"""Environment-only API settings; no .env loading or global environment mutation."""
from dataclasses import dataclass
import os
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Settings:
    model_path: str | None = None
    allowed_origins: tuple[str, ...] = ('http://localhost:3000', 'http://localhost:5173')
    max_batch_size: int = 1000

    def __post_init__(self):
        if type(self.max_batch_size) is not int or self.max_batch_size < 1:
            raise ValueError('MAX_BATCH_SIZE must be a positive integer.')
        for origin in self.allowed_origins:
            parsed = urlsplit(origin)
            if (parsed.scheme not in ('http', 'https') or not parsed.netloc
                    or parsed.path or parsed.query or parsed.fragment or parsed.username
                    or parsed.password or '*' in origin):
                raise ValueError('ALLOWED_ORIGINS must contain explicit HTTP(S) origins without paths, credentials or wildcards.')

    @classmethod
    def from_environment(cls):
        defaults = cls()
        origins = os.environ.get('ALLOWED_ORIGINS', ','.join(defaults.allowed_origins))
        try:
            maximum = int(os.environ.get('MAX_BATCH_SIZE', str(defaults.max_batch_size)))
        except ValueError:
            raise ValueError('MAX_BATCH_SIZE must be a positive integer.') from None
        return cls(model_path=os.environ.get('MODEL_PATH') or None,
                   allowed_origins=tuple(dict.fromkeys(value.strip() for value in origins.split(',') if value.strip())),
                   max_batch_size=maximum)
