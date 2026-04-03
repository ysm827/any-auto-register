"""
ChatGPT Refresh Token 注册引擎。

新实现不再沿用旧的分步补丁式注册链路，而是直接复用：
1. `ChatGPTClient.register_complete_flow()` 负责完整注册状态机
2. `OAuthClient.login_and_get_tokens()` 负责全新 OAuth + passwordless OTP 登录拿 RT

目标是让 refresh_token 模式与当前主状态机链路保持一致，不再以旧流程做兜底。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional

from core.task_runtime import TaskInterruption

from .chatgpt_client import ChatGPTClient
from .oauth import OAuthManager
from .oauth_client import OAuthClient
from .utils import (
    generate_random_birthday,
    generate_random_name,
    generate_random_password,
)

logger = logging.getLogger(__name__)


@dataclass
class RegistrationResult:
    """注册结果。"""

    success: bool
    email: str = ""
    password: str = ""
    account_id: str = ""
    workspace_id: str = ""
    access_token: str = ""
    refresh_token: str = ""
    id_token: str = ""
    session_token: str = ""
    error_message: str = ""
    logs: list | None = None
    metadata: dict | None = None
    source: str = "register"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "email": self.email,
            "password": self.password,
            "account_id": self.account_id,
            "workspace_id": self.workspace_id,
            "access_token": self.access_token[:20] + "..." if self.access_token else "",
            "refresh_token": self.refresh_token[:20] + "..." if self.refresh_token else "",
            "id_token": self.id_token[:20] + "..." if self.id_token else "",
            "session_token": self.session_token[:20] + "..." if self.session_token else "",
            "error_message": self.error_message,
            "logs": self.logs or [],
            "metadata": self.metadata or {},
            "source": self.source,
        }


@dataclass
class SignupFormResult:
    """保留旧结构，兼容外部引用。"""

    success: bool
    page_type: str = ""
    is_existing_account: bool = False
    response_data: Dict[str, Any] | None = None
    error_message: str = ""


class EmailServiceAdapter:
    """将现有 email_service 适配给 ChatGPTClient / OAuthClient 状态机。"""

    def __init__(self, email_service, email: str, log_fn: Callable[[str], None]):
        self.email_service = email_service
        self.email = email
        self.log_fn = log_fn
        self._used_codes: set[str] = set()

    def wait_for_verification_code(
        self,
        email: str,
        timeout: int = 90,
        otp_sent_at: float | None = None,
        exclude_codes=None,
    ):
        excluded = set(exclude_codes or set()) | set(self._used_codes)
        self.log_fn(f"正在等待邮箱 {email} 的验证码 ({timeout}s)...")
        code = self.email_service.get_verification_code(
            email=email,
            timeout=timeout,
            otp_sent_at=otp_sent_at,
            exclude_codes=excluded,
        )
        if code:
            code = str(code).strip()
            self._used_codes.add(code)
            self.log_fn(f"成功获取验证码: {code}")
        return code


class RefreshTokenRegistrationEngine:
    """Refresh token 注册引擎。"""

    def __init__(
        self,
        email_service,
        proxy_url: Optional[str] = None,
        callback_logger: Optional[Callable[[str], None]] = None,
        task_uuid: Optional[str] = None,
        browser_mode: str = "protocol",
        max_retries: int = 3,
        extra_config: Optional[dict] = None,
    ):
        self.email_service = email_service
        self.proxy_url = proxy_url
        self.callback_logger = callback_logger or (lambda msg: logger.info(msg))
        self.task_uuid = task_uuid
        self.browser_mode = str(browser_mode or "protocol").strip().lower() or "protocol"
        self.max_retries = max(1, int(max_retries or 1))
        self.extra_config = dict(extra_config or {})

        self.email: Optional[str] = None
        self.password: Optional[str] = None
        self.email_info: Optional[Dict[str, Any]] = None
        self.logs: list[str] = []

    def _log(self, message: str, level: str = "info"):
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        self.logs.append(log_message)

        if self.callback_logger:
            self.callback_logger(log_message)

        if level == "error":
            logger.error(log_message)
        elif level == "warning":
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _create_email(self) -> bool:
        try:
            self._log(f"正在创建 {self.email_service.service_type.value} 邮箱...")
            self.email_info = self.email_service.create_email()

            email_value = str(
                self.email
                or (self.email_info or {}).get("email")
                or ""
            ).strip()
            if not email_value:
                self._log(
                    f"创建邮箱失败: {self.email_service.service_type.value} 返回空邮箱地址",
                    "error",
                )
                return False

            if self.email_info is None:
                self.email_info = {}
            self.email_info["email"] = email_value
            self.email = email_value
            self._log(f"成功创建邮箱: {self.email}")
            return True
        except Exception as e:
            self._log(f"创建邮箱失败: {e}", "error")
            return False

    @staticmethod
    def _should_switch_to_login_after_register_failure(message: str) -> bool:
        text = str(message or "").lower()
        markers = (
            "user_already_exists",
            "account already exists",
            "please login instead",
            "add_phone",
            "add-phone",
        )
        return any(marker in text for marker in markers)

    @staticmethod
    def _should_retry(message: str) -> bool:
        text = str(message or "").lower()
        markers = (
            "tls",
            "ssl",
            "curl: (35)",
            "cloudflare",
            "authorize",
            "workspace",
            "consent",
            "otp",
            "验证码",
            "session",
            "access token",
            "refresh token",
            "network",
            "connection",
            "timeout",
            "http 400",
            "http 403",
            "registration_disallowed",
            "未获取到 authorization code",
        )
        return any(marker in text for marker in markers)

    def _read_int_config(
        self,
        primary_key: str,
        *,
        fallback_keys: tuple[str, ...] = (),
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        keys = (primary_key, *tuple(fallback_keys or ()))
        for key in keys:
            if key not in self.extra_config:
                continue
            value = self.extra_config.get(key)
            try:
                parsed = int(value)
            except Exception:
                continue
            return max(minimum, min(parsed, maximum))
        return max(minimum, min(int(default), maximum))

    def _build_chatgpt_client(self) -> ChatGPTClient:
        client = ChatGPTClient(
            proxy=self.proxy_url,
            verbose=False,
            browser_mode=self.browser_mode,
        )
        client._log = lambda msg: self._log(f"[注册链路] {msg}")
        return client

    def _build_oauth_client(self) -> OAuthClient:
        client = OAuthClient(
            self.extra_config,
            proxy=self.proxy_url,
            verbose=False,
            browser_mode=self.browser_mode,
        )
        client._log = lambda msg: self._log(f"[登录链路] {msg}")
        return client

    def _extract_account_info(self, tokens: dict[str, Any]) -> dict[str, Any]:
        id_token = str((tokens or {}).get("id_token") or "").strip()
        if not id_token:
            return {}
        manager = OAuthManager(proxy_url=self.proxy_url)
        return manager.extract_account_info(id_token)

    @staticmethod
    def _extract_workspace_id(oauth_client: OAuthClient) -> str:
        workspace_id = str(getattr(oauth_client, "last_workspace_id", "") or "").strip()
        if workspace_id:
            return workspace_id

        try:
            session_data = oauth_client._decode_oauth_session_cookie() or {}
        except Exception:
            session_data = {}

        workspaces = session_data.get("workspaces") or []
        if not workspaces:
            return ""
        return str((workspaces[0] or {}).get("id") or "").strip()

    @staticmethod
    def _extract_session_token(oauth_client: OAuthClient) -> str:
        getter = getattr(oauth_client, "_get_cookie_value", None)
        if not callable(getter):
            return ""
        return str(
            getter("__Secure-next-auth.session-token", "chatgpt.com")
            or getter("__Secure-authjs.session-token", "chatgpt.com")
            or ""
        ).strip()

    def _populate_result_from_tokens(
        self,
        result: RegistrationResult,
        tokens: dict[str, Any],
        oauth_client: OAuthClient,
        registration_message: str,
        source: str,
        register_client: ChatGPTClient,
    ) -> None:
        account_info = self._extract_account_info(tokens)
        workspace_id = self._extract_workspace_id(oauth_client)
        session_token = self._extract_session_token(oauth_client)

        result.success = True
        result.email = self.email or ""
        result.password = self.password or ""
        result.access_token = str(tokens.get("access_token") or "").strip()
        result.refresh_token = str(tokens.get("refresh_token") or "").strip()
        result.id_token = str(tokens.get("id_token") or "").strip()
        result.account_id = str(
            tokens.get("account_id")
            or account_info.get("account_id")
            or ""
        ).strip()
        result.workspace_id = workspace_id
        result.session_token = session_token
        result.source = source
        result.metadata = {
            "email_service": self.email_service.service_type.value,
            "proxy_used": self.proxy_url,
            "registered_at": datetime.now().isoformat(),
            "registration_message": registration_message,
            "registration_flow": "chatgpt_client.register_complete_flow",
            "token_flow": "oauth_client.login_and_get_tokens",
            "token_login_mode": "passwordless",
            "browser_mode": self.browser_mode,
            "device_id": getattr(register_client, "device_id", ""),
            "impersonate": getattr(register_client, "impersonate", ""),
            "user_agent": getattr(register_client, "ua", ""),
            "workspace_id": workspace_id,
            "account_claims_email": account_info.get("email", ""),
        }

    def run(self) -> RegistrationResult:
        result = RegistrationResult(success=False, logs=self.logs)
        last_error = ""
        fixed_email = str(self.email or "").strip()
        register_otp_wait_seconds = self._read_int_config(
            "chatgpt_register_otp_wait_seconds",
            fallback_keys=("chatgpt_otp_wait_seconds",),
            default=600,
            minimum=30,
            maximum=3600,
        )
        register_otp_resend_wait_seconds = self._read_int_config(
            "chatgpt_register_otp_resend_wait_seconds",
            fallback_keys=("chatgpt_register_otp_wait_seconds", "chatgpt_otp_wait_seconds"),
            default=300,
            minimum=30,
            maximum=3600,
        )

        try:
            for attempt in range(self.max_retries):
                registration_message = ""
                source = "register"

                try:
                    if attempt == 0:
                        self._log("=" * 60)
                        self._log("ChatGPT RT 全新主链路启动")
                        self._log(f"请求模式: {self.browser_mode}")
                        self._log("实现策略: 注册状态机 + 全新 OAuth session + passwordless OTP")
                        self._log("=" * 60)
                    else:
                        self._log(f"整流程重试 {attempt + 1}/{self.max_retries} ...")
                        time.sleep(1)

                    # 用户未显式指定邮箱时，每轮都必须使用本轮新购邮箱，
                    # 避免重试时误复用上一轮邮箱导致收不到验证码。
                    if not fixed_email:
                        self.email = None

                    self._log("1. 创建邮箱...")
                    if not self._create_email():
                        last_error = "创建邮箱失败"
                        if attempt < self.max_retries - 1:
                            self._log(f"{last_error}，准备整流程重试", "warning")
                            continue
                        result.error_message = last_error
                        return result

                    result.email = self.email or ""
                    self.password = self.password or generate_random_password(16)
                    result.password = self.password

                    first_name, last_name = generate_random_name()
                    birthdate = generate_random_birthday()
                    self._log(f"邮箱: {result.email}")
                    self._log(f"密码: {self.password}")
                    self._log(f"注册信息: {first_name} {last_name}, 生日: {birthdate}")
                    self._log("流程策略: 注册阶段到 about_you 即停，改由 OAuth 新会话补全资料")
                    self._log(
                        "验证码等待策略: "
                        f"register_wait={register_otp_wait_seconds}s, "
                        f"register_resend_wait={register_otp_resend_wait_seconds}s, "
                        "oauth_wait=读取 OAuthClient 配置（默认600s）"
                    )

                    email_adapter = EmailServiceAdapter(
                        self.email_service,
                        result.email,
                        self._log,
                    )

                    register_client = self._build_chatgpt_client()
                    self._log("2. 执行注册状态机（interrupt 模式：不在注册阶段提交 about_you）...")
                    registered, registration_message = register_client.register_complete_flow(
                        result.email,
                        self.password,
                        first_name,
                        last_name,
                        birthdate,
                        email_adapter,
                        stop_before_about_you_submission=True,
                        otp_wait_timeout=register_otp_wait_seconds,
                        otp_resend_wait_timeout=register_otp_resend_wait_seconds,
                    )

                    if not registered:
                        if not self._should_switch_to_login_after_register_failure(registration_message):
                            last_error = f"注册状态机失败: {registration_message}"
                            if attempt < self.max_retries - 1 and self._should_retry(last_error):
                                self._log(f"{last_error}，准备整流程重试", "warning")
                                continue
                            result.error_message = last_error
                            return result

                        source = "login"
                        self._log(
                            "注册阶段命中可恢复终态，直接切换到全新 OAuth + passwordless 登录链路",
                            "warning",
                        )
                        self._log(f"切换原因: {registration_message}")
                    else:
                        if registration_message == "pending_about_you_submission":
                            self._log("注册状态机已推进至 about_you，符合预期。下一步进入 OAuth 会话补全资料")
                        else:
                            self._log(
                                "注册状态机返回成功但未停在 about_you。"
                                "将继续进入 OAuth 会话，按状态机实际返回推进。"
                            )

                    oauth_client = self._build_oauth_client()
                    self._log("3. 新开 OAuth session，按 screen_hint=login + passwordless OTP 登录...")
                    self._log("4. 若命中 about_you，则在 OAuth 会话内提交姓名+生日，再继续 workspace/token")
                    tokens = oauth_client.login_and_get_tokens(
                        result.email,
                        self.password,
                        device_id="",
                        user_agent=getattr(register_client, "ua", None),
                        sec_ch_ua=getattr(register_client, "sec_ch_ua", None),
                        impersonate=getattr(register_client, "impersonate", None),
                        skymail_client=email_adapter,
                        prefer_passwordless_login=True,
                        allow_phone_verification=False,
                        force_new_browser=True,
                        complete_about_you_if_needed=True,
                        first_name=first_name,
                        last_name=last_name,
                        birthdate=birthdate,
                        login_source=(
                            "existing_account_recovery"
                            if source == "login"
                            else "post_register_workspace_recovery"
                        ),
                    )
                    if not tokens:
                        last_error = oauth_client.last_error or "OAuth 登录状态机失败"
                        if attempt < self.max_retries - 1 and self._should_retry(last_error):
                            self._log(f"{last_error}，准备整流程重试", "warning")
                            continue
                        result.error_message = last_error
                        return result

                    self._populate_result_from_tokens(
                        result=result,
                        tokens=tokens,
                        oauth_client=oauth_client,
                        registration_message=registration_message or "register_complete_flow:ok",
                        source=source,
                        register_client=register_client,
                    )

                    self._log("5. 主链路完成")
                    self._log(f"Account ID: {result.account_id}")
                    self._log(f"Workspace ID: {result.workspace_id}")
                    self._log("=" * 60)
                    return result

                except TaskInterruption:
                    raise
                except Exception as e:
                    last_error = str(e)
                    if attempt < self.max_retries - 1 and self._should_retry(last_error):
                        self._log(f"本轮出现异常，准备整流程重试: {last_error}", "warning")
                        continue
                    raise

            result.error_message = last_error or "注册失败"
            return result

        except TaskInterruption:
            raise
        except Exception as e:
            self._log(f"RT 注册主链路异常: {e}", "error")
            result.error_message = str(e)
            return result

    def save_to_database(self, result: RegistrationResult) -> bool:
        """保留旧接口，占位返回。"""
        return bool(result and result.success)
