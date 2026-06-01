"""Custom LangGraph authentication with JWT and local development modes."""

from __future__ import annotations

import base64
import binascii
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import anyio.to_thread

try:  # pragma: no cover - dependency availability is environment-specific.
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric import ec, padding, rsa
    from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
    from cryptography.hazmat.primitives.hashes import SHA256, SHA384, SHA512
except Exception:  # pragma: no cover - validated at runtime when signatures are required.
    InvalidSignature = None  # type: ignore[assignment,misc]
    ec = None  # type: ignore[assignment]
    padding = None  # type: ignore[assignment]
    rsa = None  # type: ignore[assignment]
    encode_dss_signature = None  # type: ignore[assignment]
    SHA256 = SHA384 = SHA512 = None  # type: ignore[assignment,misc]

from backend.app.v1.utils.langsmith import public_auth_metadata
from backend.app.v1.utils.azure_key_vault import resolve_env_secret
from backend.app.v1.utils.helper import hash_identifier, token_fingerprint

_LangGraphAuth = None  # type: ignore[assignment,misc]


class AuthValidationError(ValueError):
    """Raised when credentials cannot be authenticated."""


class AuthConfigurationError(RuntimeError):
    """Raised when auth is configured in an unsafe or incomplete way."""


class _FallbackHTTPError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class _FallbackExceptions:
    HTTPException = _FallbackHTTPError


class _FallbackAuth:
    """Tiny stand-in so local imports work without langgraph_sdk installed."""

    exceptions = _FallbackExceptions

    class _Types:
        MinimalUserDict = dict[str, Any]

    types = _Types

    def authenticate(self, fn: Any) -> Any:
        return fn


Auth: Any = _LangGraphAuth if _LangGraphAuth is not None else _FallbackAuth


@dataclass(frozen=True)
class AuthConfig:
    mode: str = "jwt"
    issuer: str | None = None
    audience: str | None = None
    allowed_audiences: tuple[str, ...] = ()
    tenant: str | None = None
    jwks_url: str | None = None
    jwks_json: str | None = None
    algorithms: tuple[str, ...] = ("RS256",)
    required_scopes: tuple[str, ...] = ()
    allow_unverified_jwt: bool = False
    clock_skew_seconds: int = 60
    jwks_cache_seconds: int = 300
    dev_subject: str = "local-developer"
    dev_tenant: str = "local"
    dev_scopes: tuple[str, ...] = ("threads:read", "threads:write", "runs:read", "runs:write")

    @classmethod
    def from_env(cls, prefix: str = "AGENT_AUTH_") -> AuthConfig:
        config = cls(
            mode=os.getenv(prefix + "MODE", "jwt").strip().lower(),
            issuer=_empty_to_none(os.getenv(prefix + "ISSUER") or os.getenv("ENTRA_ISSUER")),
            audience=_empty_to_none(os.getenv(prefix + "AUDIENCE") or os.getenv("ENTRA_AUDIENCE")),
            allowed_audiences=_csv(
                os.getenv(prefix + "ALLOWED_AUDIENCES") or os.getenv("ENTRA_ALLOWED_AUDIENCES", "")
            ),
            tenant=_empty_to_none(os.getenv(prefix + "TENANT") or os.getenv("ENTRA_TENANT_ID")),
            jwks_url=_empty_to_none(os.getenv(prefix + "JWKS_URL") or os.getenv("ENTRA_JWKS_URL")),
            jwks_json=_empty_to_none(resolve_env_secret(prefix + "JWKS_JSON")),
            algorithms=_csv(os.getenv(prefix + "ALGORITHMS", "RS256")),
            required_scopes=_csv(
                os.getenv(prefix + "REQUIRED_SCOPES") or os.getenv("ENTRA_REQUIRED_SCOPES", "")
            ),
            allow_unverified_jwt=_truthy(os.getenv(prefix + "ALLOW_UNVERIFIED_JWT")),
            clock_skew_seconds=int(os.getenv(prefix + "CLOCK_SKEW_SECONDS", "60")),
            jwks_cache_seconds=int(os.getenv(prefix + "JWKS_CACHE_SECONDS", "300")),
            dev_subject=os.getenv(prefix + "DEV_SUBJECT", "local-developer"),
            dev_tenant=os.getenv(prefix + "DEV_TENANT", "local"),
            dev_scopes=_csv(
                os.getenv(
                    prefix + "DEV_SCOPES",
                    "threads:read,threads:write,runs:read,runs:write",
                )
            ),
        )
        config.validate_for_environment()
        return config

    def validate_for_environment(self) -> None:
        """Hard-guard dev mode and unsafe-JWT toggles in non-dev environments,
        raising ``AuthConfigurationError`` on an unsafe or incomplete config."""

        app_env = (os.getenv("APP_ENV") or "local").strip().lower()
        dev_envs = {"local", "dev"}
        if self.mode == "dev" and app_env not in dev_envs:
            raise AuthConfigurationError(
                "AGENT_AUTH_MODE=dev is only allowed when APP_ENV is 'local' or 'dev' "
                f"(APP_ENV={app_env!r})."
            )
        if self.mode == "jwt" and app_env not in dev_envs:
            if not (self.jwks_url or self.jwks_json):
                raise AuthConfigurationError(
                    "JWT signature validation requires JWKS configuration in non-dev "
                    "environments (set AGENT_AUTH_JWKS_URL or ENTRA_JWKS_URL)."
                )
            if not self.issuer:
                raise AuthConfigurationError(
                    "JWT issuer validation is required in non-dev environments "
                    "(set AGENT_AUTH_ISSUER or ENTRA_ISSUER)."
                )
            if not _accepted_audiences(self):
                raise AuthConfigurationError(
                    "JWT audience validation is required in non-dev environments "
                    "(set AGENT_AUTH_AUDIENCE or AGENT_AUTH_ALLOWED_AUDIENCES)."
                )
            if not self.required_scopes:
                raise AuthConfigurationError(
                    "JWT scope validation is required in non-dev environments "
                    "(set AGENT_AUTH_REQUIRED_SCOPES or ENTRA_REQUIRED_SCOPES)."
                )
            if self.jwks_url:
                parsed_url = urllib.parse.urlparse(self.jwks_url)
                if parsed_url.scheme != "https":
                    raise AuthConfigurationError(
                        "JWT JWKS URL must use https in non-dev environments."
                    )


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    subject: str = field(repr=False)
    auth_mode: str
    issuer: str | None = field(default=None, repr=False)
    audience: str | None = field(default=None, repr=False)
    tenant: str | None = field(default=None, repr=False)
    scopes: tuple[str, ...] = ()
    groups: tuple[str, ...] = ()
    permissions: tuple[str, ...] = ()
    claims: Mapping[str, Any] = field(default_factory=dict, repr=False)
    token_fingerprint: str | None = None
    authenticated: bool = True

    @property
    def identity(self) -> str:
        hashed_subject = hash_identifier(self.subject) or "anonymous"
        return f"user:{hashed_subject}"

    def to_langgraph_user(self) -> dict[str, Any]:
        metadata = public_auth_metadata(self)
        return {
            "identity": self.identity,
            "is_authenticated": self.authenticated,
            "permissions": list(self.permissions),
            "groups": list(self.groups),
            "metadata": metadata,
        }


class JwtAuthenticator:
    def __init__(self, config: AuthConfig | None = None) -> None:
        # config is None for the module-level singleton; env config is resolved
        # lazily on first call and memoized. Tests pass an explicit AuthConfig.
        self._explicit_config = config
        self.config: AuthConfig = config  # type: ignore[assignment]
        self._cached_env_config: AuthConfig | None = None
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_expires = 0.0
        self._jwks_lock = threading.Lock()

    def _resolve_config(self) -> AuthConfig:
        if self._explicit_config is not None:
            self.config = self._explicit_config
            return self._explicit_config
        if self._cached_env_config is not None:
            self.config = self._cached_env_config
            return self._cached_env_config
        cfg = AuthConfig.from_env()
        self._cached_env_config = cfg
        self.config = cfg
        return cfg

    def authenticate_authorization(
        self,
        authorization: str | None,
        headers: Mapping[str | bytes, str | bytes] | None = None,
    ) -> AuthenticatedPrincipal:
        self._resolve_config()
        if self.config.mode in {"dev", "mock", "local"}:
            return self._dev_principal(authorization, headers)
        if self.config.mode in {"off", "disabled", "none"}:
            raise AuthConfigurationError("Authentication must not be disabled in this handler")

        token = _extract_bearer_token(authorization)
        header, claims, signing_input, signature = _decode_jwt(token)
        self._validate_claims(claims)
        self._validate_signature(header, signing_input, signature)

        subject = (
            _claim_str(claims, "oid") or _claim_str(claims, "sub") or _claim_str(claims, "uid")
        )
        if not subject:
            raise AuthValidationError("JWT missing subject claim")

        scopes = _claim_scopes(claims)
        groups = _claim_groups(claims)
        missing_scopes = set(self.config.required_scopes) - set(scopes)
        if missing_scopes:
            raise AuthValidationError("JWT missing required scope")

        tenant = _claim_str(claims, "tid") or _claim_str(claims, "tenant_id")
        audience = _claim_audience(claims)
        return AuthenticatedPrincipal(
            subject=subject,
            auth_mode="jwt",
            issuer=_claim_str(claims, "iss"),
            audience=",".join(audience),
            tenant=tenant,
            scopes=tuple(scopes),
            groups=tuple(groups),
            permissions=tuple(scopes),
            claims=_safe_claims(claims),
            token_fingerprint=token_fingerprint(token),
        )

    def _dev_principal(
        self,
        authorization: str | None,
        headers: Mapping[str | bytes, str | bytes] | None,
    ) -> AuthenticatedPrincipal:
        token = None
        if authorization:
            try:
                token = _extract_bearer_token(authorization)
            except AuthValidationError:
                token = authorization
        subject = _header_value(headers or {}, "x-dev-subject") or self.config.dev_subject
        tenant = _header_value(headers or {}, "x-dev-tenant") or self.config.dev_tenant
        scopes = _csv(_header_value(headers or {}, "x-dev-scopes")) or self.config.dev_scopes
        groups = _csv(_header_value(headers or {}, "x-dev-groups"))
        return AuthenticatedPrincipal(
            subject=subject,
            auth_mode="dev",
            issuer="local-dev",
            audience="local-dev",
            tenant=tenant,
            scopes=scopes,
            groups=groups,
            permissions=scopes,
            claims={"mode": "dev"},
            token_fingerprint=token_fingerprint(token),
        )

    def _validate_claims(self, claims: Mapping[str, Any]) -> None:
        now = int(time.time())
        skew = self.config.clock_skew_seconds

        exp = _claim_int(claims, "exp")
        app_env = (os.getenv("APP_ENV") or "local").strip().lower()
        if exp is None and app_env not in {"local", "dev"}:
            raise AuthValidationError("JWT missing exp claim")
        if exp is not None and now > exp + skew:
            raise AuthValidationError("JWT expired")

        nbf = _claim_int(claims, "nbf")
        if nbf is not None and now + skew < nbf:
            raise AuthValidationError("JWT not yet valid")

        issuer = _claim_str(claims, "iss")
        if self.config.issuer and issuer != self.config.issuer:
            raise AuthValidationError("JWT issuer mismatch")

        accepted_audiences = _accepted_audiences(self.config)
        if accepted_audiences and not set(_claim_audience(claims)).intersection(accepted_audiences):
            raise AuthValidationError("JWT audience mismatch")

        tenant = _claim_str(claims, "tid") or _claim_str(claims, "tenant_id")
        if self.config.tenant and tenant != self.config.tenant:
            raise AuthValidationError("JWT tenant mismatch")

    def _validate_signature(
        self,
        header: Mapping[str, Any],
        signing_input: bytes,
        signature: bytes,
    ) -> None:
        alg = _claim_str(header, "alg")
        if not alg or alg == "none" or alg not in self.config.algorithms:
            raise AuthValidationError("JWT algorithm is not allowed")

        if not (self.config.jwks_url or self.config.jwks_json):
            app_env = (os.getenv("APP_ENV") or "local").strip().lower()
            if self.config.allow_unverified_jwt and app_env in {"local", "dev"}:
                return
            raise AuthConfigurationError("JWT signature validation requires JWKS configuration")
        required_crypto = (
            InvalidSignature,
            ec,
            padding,
            rsa,
            encode_dss_signature,
            SHA256,
            SHA384,
            SHA512,
        )
        if any(item is None for item in required_crypto):
            raise AuthConfigurationError("JWT signature validation requires cryptography")

        key = self._select_jwk(header)
        public_key = _jwk_to_public_key(key)
        verifier = _verifier_for_alg(alg, signature)
        try:
            if alg.startswith("ES"):
                public_key.verify(verifier["signature"], signing_input, verifier["hash"])
            else:
                public_key.verify(
                    verifier["signature"],
                    signing_input,
                    verifier["padding"],
                    verifier["hash"],
                )
        except InvalidSignature as exc:
            raise AuthValidationError("JWT signature invalid") from exc

    def _select_jwk(self, header: Mapping[str, Any]) -> Mapping[str, Any]:
        kid = _claim_str(header, "kid")
        jwks = self._load_jwks()
        keys = jwks.get("keys")
        if not isinstance(keys, list) or not keys:
            raise AuthConfigurationError("JWKS must contain a non-empty keys array")
        if kid:
            for key in keys:
                if isinstance(key, Mapping) and key.get("kid") == kid:
                    return key
            raise AuthValidationError("JWT key id not found in JWKS")
        if len(keys) == 1 and isinstance(keys[0], Mapping):
            return keys[0]
        raise AuthValidationError("JWT key id is required when JWKS has multiple keys")

    def _load_jwks(self) -> Mapping[str, Any]:
        if self.config.jwks_json:
            jwks = json.loads(self.config.jwks_json)
            if not isinstance(jwks, Mapping):
                raise AuthConfigurationError("JWKS JSON must be an object")
            return jwks

        now = time.time()
        cached = self._jwks_cache
        if cached and now < self._jwks_cache_expires:
            return cached

        with self._jwks_lock:
            now = time.time()
            cached = self._jwks_cache
            if cached and now < self._jwks_cache_expires:
                return cached

            if not self.config.jwks_url:
                raise AuthConfigurationError("JWKS URL is not configured")

            parsed_url = urllib.parse.urlparse(self.config.jwks_url)
            app_env = (os.getenv("APP_ENV") or "local").strip().lower()
            if app_env in {"local", "dev"}:
                allowed_schemes = {"http", "https"}
            else:
                allowed_schemes = {"https"}
            if parsed_url.scheme not in allowed_schemes:
                raise AuthConfigurationError(
                    "JWKS URL must use https"
                    if app_env not in {"local", "dev"}
                    else "JWKS URL must use http or https"
                )

            with urllib.request.urlopen(self.config.jwks_url, timeout=5) as response:  # noqa: S310
                jwks = json.loads(response.read().decode("utf-8"))
            if not isinstance(jwks, Mapping):
                raise AuthConfigurationError("JWKS response must be an object")
            self._jwks_cache = dict(jwks)
            self._jwks_cache_expires = now + self.config.jwks_cache_seconds
            return self._jwks_cache


def _extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise AuthValidationError("Missing Authorization header")
    scheme, _, token = authorization.strip().partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise AuthValidationError("Authorization must use Bearer scheme")
    return token.strip()


def extract_bearer_token(authorization: str | None) -> str:
    """Return the Bearer token from an Authorization header."""

    return _extract_bearer_token(authorization)


def _decode_jwt(token: str) -> tuple[Mapping[str, Any], Mapping[str, Any], bytes, bytes]:
    parts = token.split(".")
    if len(parts) != 3:
        raise AuthValidationError("Bearer token is not a JWT")
    try:
        header = _json_b64url(parts[0])
        claims = _json_b64url(parts[1])
        signature = _b64url_decode(parts[2])
    except AuthValidationError:
        raise
    except (binascii.Error, json.JSONDecodeError, UnicodeDecodeError, ValueError) as exc:
        raise AuthValidationError("Bearer token is not a valid JWT") from exc
    if not isinstance(header, Mapping) or not isinstance(claims, Mapping):
        raise AuthValidationError("JWT header and payload must be objects")
    signing_input = f"{parts[0]}.{parts[1]}".encode("ascii")
    return header, claims, signing_input, signature


def _json_b64url(value: str) -> Any:
    return json.loads(_b64url_decode(value).decode("utf-8"))


def _b64url_decode(value: str) -> bytes:
    padding_length = (-len(value)) % 4
    return base64.urlsafe_b64decode(value + ("=" * padding_length))


def _claim_str(claims: Mapping[str, Any], key: str) -> str | None:
    value = claims.get(key)
    return value if isinstance(value, str) else None


def _claim_int(claims: Mapping[str, Any], key: str) -> int | None:
    value = claims.get(key)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return None


def _claim_audience(claims: Mapping[str, Any]) -> tuple[str, ...]:
    audience = claims.get("aud")
    if isinstance(audience, str):
        return (audience,)
    if isinstance(audience, Sequence):
        return tuple(str(item) for item in audience)
    return ()


def _claim_scopes(claims: Mapping[str, Any]) -> tuple[str, ...]:
    scopes: set[str] = set()
    for key in ("scope", "scp"):
        value = claims.get(key)
        if isinstance(value, str):
            scopes.update(value.split())
    roles = claims.get("roles")
    if isinstance(roles, Sequence) and not isinstance(roles, str | bytes | bytearray):
        scopes.update(str(role) for role in roles)
    return tuple(sorted(scopes))


def _claim_groups(claims: Mapping[str, Any]) -> tuple[str, ...]:
    groups: set[str] = set()
    for key in ("groups", "group_ids"):
        value = claims.get(key)
        if isinstance(value, str):
            groups.add(value)
        elif isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray):
            groups.update(str(group) for group in value if str(group).strip())
    return tuple(sorted(groups))


def _safe_claims(claims: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {
        "aud",
        "exp",
        "groups",
        "iat",
        "iss",
        "nbf",
        "roles",
        "scp",
        "scope",
        "tid",
        "tenant_id",
    }
    return {key: claims[key] for key in allowed if key in claims}


def _jwk_to_public_key(jwk: Mapping[str, Any]) -> Any:
    kty = jwk.get("kty")
    if kty == "RSA":
        n = int.from_bytes(_b64url_decode(str(jwk["n"])), "big")
        e = int.from_bytes(_b64url_decode(str(jwk["e"])), "big")
        return rsa.RSAPublicNumbers(e, n).public_key()
    if kty == "EC":
        curve_name = jwk.get("crv")
        curves = {"P-256": ec.SECP256R1(), "P-384": ec.SECP384R1(), "P-521": ec.SECP521R1()}
        if curve_name not in curves:
            raise AuthConfigurationError("Unsupported EC curve in JWKS")
        x = int.from_bytes(_b64url_decode(str(jwk["x"])), "big")
        y = int.from_bytes(_b64url_decode(str(jwk["y"])), "big")
        return ec.EllipticCurvePublicNumbers(x, y, curves[curve_name]).public_key()
    raise AuthConfigurationError("Unsupported JWKS key type")


def _verifier_for_alg(alg: str, signature: bytes) -> dict[str, Any]:
    hashes = {
        "RS256": SHA256(),
        "RS384": SHA384(),
        "RS512": SHA512(),
        "ES256": SHA256(),
        "ES384": SHA384(),
        "ES512": SHA512(),
    }
    if alg not in hashes:
        raise AuthValidationError("Unsupported JWT algorithm")
    if alg.startswith("ES"):
        size = len(signature) // 2
        der_signature = encode_dss_signature(
            int.from_bytes(signature[:size], "big"),
            int.from_bytes(signature[size:], "big"),
        )
        return {"signature": der_signature, "hash": ec.ECDSA(hashes[alg])}
    return {"signature": signature, "padding": padding.PKCS1v15(), "hash": hashes[alg]}


def _header_value(headers: Mapping[str | bytes, str | bytes], name: str) -> str | None:
    wanted = name.lower()
    for key, value in headers.items():
        key_text = key.decode("latin1") if isinstance(key, bytes) else str(key)
        if key_text.lower() == wanted:
            return value.decode("latin1") if isinstance(value, bytes) else str(value)
    return None


def _accepted_audiences(config: AuthConfig) -> set[str]:
    accepted = set(config.allowed_audiences)
    if config.audience:
        accepted.add(config.audience)
    return accepted


def _csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    import re

    return tuple(item.strip() for item in re.split(r"[\s,]+", value) if item.strip())


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _empty_to_none(value: str | None) -> str | None:
    if value is None or not value.strip():
        return None
    return value.strip()


authenticator = JwtAuthenticator()
auth = Auth()
my_auth = auth


@auth.authenticate  # type: ignore[untyped-decorator]
async def authenticate(
    authorization: str | None = None,
    headers: Mapping[str | bytes, str | bytes] | None = None,
) -> dict[str, Any]:
    """LangGraph SDK authentication entry point."""

    try:
        # authenticate_authorization can block on a cold-cache JWKS fetch; offload
        # to a worker thread so the event loop stays responsive (JWKS cache is locked).
        principal = await anyio.to_thread.run_sync(
            authenticator.authenticate_authorization, authorization, headers
        )
    except AuthConfigurationError as exc:
        raise auth.exceptions.HTTPException(status_code=500, detail=str(exc)) from exc
    except AuthValidationError as exc:
        detail = "Unauthorized"
        if authorization:
            detail = f"Unauthorized:{token_fingerprint(authorization)}"
        raise auth.exceptions.HTTPException(status_code=401, detail=detail) from exc
    return principal.to_langgraph_user()


# Thread ownership filters: bind every thread (and its runs) to the caller's
# identity so users only access their own conversations. Guarded by hasattr so
# the _FallbackAuth path stays import-safe.


def _ctx_identity(ctx: Any) -> str | None:
    user = getattr(ctx, "user", None)
    if isinstance(user, Mapping):
        value = user.get("identity")
    else:
        value = getattr(user, "identity", None)
    return str(value) if value else None


async def on_thread_create(ctx: Any, value: dict[str, Any]) -> dict[str, Any] | bool:
    """Stamp the owner identity on thread creation and return a filter."""

    owner = _ctx_identity(ctx)
    if owner is None:
        return False
    if isinstance(value, dict):
        metadata = value.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            value["metadata"] = metadata
        metadata["owner"] = owner
    return {"owner": owner}


async def on_thread_access(ctx: Any, value: Any) -> dict[str, Any] | bool:
    """Restrict reads/updates/deletes to threads owned by the caller."""

    owner = _ctx_identity(ctx)
    if owner is None:
        return False
    return {"owner": owner}


if hasattr(auth, "on"):  # pragma: no branch - depends on SDK availability
    auth.on.threads.create(on_thread_create)
    auth.on.threads.read(on_thread_access)
    auth.on.threads.search(on_thread_access)
    auth.on.threads.update(on_thread_access)
    auth.on.threads.delete(on_thread_access)
    auth.on.threads.create_run(on_thread_access)
