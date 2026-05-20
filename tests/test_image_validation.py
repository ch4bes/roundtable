"""Tests for image path validation (Issue #32 — Local File Disclosure).

The OllamaClient should refuse to read arbitrary files when the user
supplies a path via ``--image``.  Only files with a whitelisted image
extension **and** a matching magic-byte header are allowed through.
"""

from pathlib import Path

import pytest

from core.ollama_client import _is_valid_image_path, _ALLOWED_IMAGE_EXTENSIONS


# ===== helper: create a minimal valid image file ============================


def _write_png(tmp_path: Path, name: str = "img.png") -> Path:
    """Create a minimal PNG file with valid header bytes."""
    p = tmp_path / name
    # 8-byte PNG header + 1 byte payload to satisfy min_len checks
    p.write_bytes(b"\x89PNG\r\n\x1a\n\x00")
    return p


def _write_jpg(tmp_path: Path, name: str = "img.jpg") -> Path:
    p = tmp_path / name
    p.write_bytes(b"\xff\xd8\xff\xe0")
    return p


def _write_jpeg(tmp_path: Path, name: str = "img.jpeg") -> Path:
    return _write_jpg(tmp_path, name)


def _write_gif(tmp_path: Path, name: str = "img.gif") -> Path:
    p = tmp_path / name
    p.write_bytes(b"GIF89a\x01\x00\x01\x00")
    return p


def _write_bmp(tmp_path: Path, name: str = "img.bmp") -> Path:
    p = tmp_path / name
    p.write_bytes(b"BM")
    return p


def _write_tiff(tmp_path: Path, name: str = "img.tiff") -> Path:
    p = tmp_path / name
    p.write_bytes(b"II\x2A\x00")
    return p


def _write_webp(tmp_path: Path, name: str = "img.webp") -> Path:
    p = tmp_path / name
    # RIFF<header size>WEBP
    p.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
    return p


# ===== allowed-extensions sanity check =====================================


class TestAllowedExtensions:
    def test_png_allowed(self):
        assert ".png" in _ALLOWED_IMAGE_EXTENSIONS

    def test_jpg_allowed(self):
        assert ".jpg" in _ALLOWED_IMAGE_EXTENSIONS

    def test_jpeg_allowed(self):
        assert ".jpeg" in _ALLOWED_IMAGE_EXTENSIONS

    def test_webp_allowed(self):
        assert ".webp" in _ALLOWED_IMAGE_EXTENSIONS

    def test_gif_allowed(self):
        assert ".gif" in _ALLOWED_IMAGE_EXTENSIONS

    def test_bmp_allowed(self):
        assert ".bmp" in _ALLOWED_IMAGE_EXTENSIONS

    def test_tiff_allowed(self):
        assert ".tiff" in _ALLOWED_IMAGE_EXTENSIONS

    def test_txt_not_allowed(self):
        assert ".txt" not in _ALLOWED_IMAGE_EXTENSIONS

    def test_json_not_allowed(self):
        assert ".json" not in _ALLOWED_IMAGE_EXTENSIONS

    def test_python_not_allowed(self):
        assert ".py" not in _ALLOWED_IMAGE_EXTENSIONS


# ===== extension checks ======================================================


class TestExtensionWhitelist:
    def test_rejects_txt(self, tmp_path):
        p = tmp_path / "file.txt"
        p.write_text("hello")
        ok, reason = _is_valid_image_path(p)
        assert not ok
        assert "not in allowed set" in reason

    def test_rejects_json(self, tmp_path):
        p = tmp_path / "file.json"
        p.write_text("{}")
        ok, reason = _is_valid_image_path(p)
        assert not ok

    def test_rejects_no_extension(self, tmp_path):
        p = tmp_path / "Makefile"
        p.write_text("all:")
        ok, reason = _is_valid_image_path(p)
        assert not ok
        assert "not in allowed set" in reason

    def test_rejects_uppercase_extension(self, tmp_path):
        # .PNG (uppercase) should still be accepted (case-insensitive)
        p = tmp_path / "file.PNG"
        p.write_bytes(b"\x89PNG\r\n\x1a\n\x00")
        ok, _ = _is_valid_image_path(p)
        assert ok


# ===== magic-byte checks =====================================================


class TestMagicBytes:
    def test_valid_png(self, tmp_path):
        p = _write_png(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_png2(self, tmp_path):
        p = _write_png(tmp_path, "file2.png")
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_jpg(self, tmp_path):
        p = _write_jpg(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_jpeg(self, tmp_path):
        p = _write_jpeg(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_gif(self, tmp_path):
        p = _write_gif(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_bmp(self, tmp_path):
        p = _write_bmp(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_tiff(self, tmp_path):
        p = _write_tiff(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_tiff_big_endian(self, tmp_path):
        p = tmp_path / "big.tiff"
        p.write_bytes(b"MM\x00\x2A")
        ok, _ = _is_valid_image_path(p)
        assert ok

    def test_valid_webp(self, tmp_path):
        p = _write_webp(tmp_path)
        ok, _ = _is_valid_image_path(p)
        assert ok


# ===== forged files (image extension but wrong content) ======================


class TestForgedFiles:
    """Files with a valid extension but wrong content should be rejected."""

    def test_png_extension_with_text_content(self, tmp_path):
        p = tmp_path / "passwd.png"
        p.write_text("root:x:0:0:root:/root:/bin/bash")
        ok, reason = _is_valid_image_path(p)
        assert not ok
        assert "header" in reason.lower()

    def test_png_extension_with_json_content(self, tmp_path):
        p = tmp_path / "config.json"
        # Actually we want a .png extension
        p = tmp_path / "config.png"
        p.write_text('{"key": "value"}')
        ok, reason = _is_valid_image_path(p)
        assert not ok

    def test_jpg_extension_with_python_code(self, tmp_path):
        p = tmp_path / "script.jpg"
        p.write_text("import os; os.system('ls')")
        ok, reason = _is_valid_image_path(p)
        assert not ok

    def test_webp_extension_with_markdown(self, tmp_path):
        p = tmp_path / "readme.webp"
        p.write_text("# Hello World\nThis is markdown")
        ok, reason = _is_valid_image_path(p)
        assert not ok

    def test_gif_extension_with_csv(self, tmp_path):
        p = tmp_path / "data.gif"
        p.write_text("a,b,c\n1,2,3")
        ok, reason = _is_valid_image_path(p)
        assert not ok


# ===== system-file simulation ================================================


class TestSystemFileProtection:
    """Simulate attempts to read system-critical files."""

    def test_etc_passwd_as_png(self, tmp_path):
        """Renaming /etc/passwd to .png should still fail (wrong magic bytes)."""
        p = tmp_path / "passwd.png"
        # Simulate passwd file contents
        p.write_bytes(b"root:x:0:0:root:/root:/bin/bash\n")
        ok, _ = _is_valid_image_path(p)
        assert not ok

    def test_ssh_key_as_jpg(self, tmp_path):
        p = tmp_path / "id_rsa.jpg"
        p.write_bytes(b"-----BEGIN OPENSSH PRIVATE KEY-----\ndata")
        ok, _ = _is_valid_image_path(p)
        assert not ok

    def test_dotenv_as_tiff(self, tmp_path):
        p = tmp_path / ".env.tiff"
        p.write_text("SECRET_KEY=abc123\nDB_PASS=xyz")
        ok, _ = _is_valid_image_path(p)
        assert not ok

    def test_etc_shadow_as_bmp(self, tmp_path):
        p = tmp_path / "shadow.bmp"
        p.write_bytes(b"root:$6$rounds=65536$...")
        ok, _ = _is_valid_image_path(p)
        assert not ok

    def test_config_json_as_png(self, tmp_path):
        p = tmp_path / "config.png"
        p.write_text('{"database": {"password": "secret"}}')
        ok, _ = _is_valid_image_path(p)
        assert not ok


# ===== real-world image files ================================================


class TestRealWorldImages:
    """Use actual image files to confirm they pass validation."""

    def test_real_png_from_tests(self, tmp_path):
        # Create a proper PNG with IHDR chunk
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x01\x00\x00\x00\x00\x7c\x18\x95\xf6"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x06\x00"
            b"\x01\r\n-\xb4"
            b"\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        p = tmp_path / "real.png"
        p.write_bytes(png_data)
        ok, _ = _is_valid_image_path(p)
        assert ok


# ===== edge cases ===========================================================


class TestEdgeCases:
    def test_empty_file_with_png_extension(self, tmp_path):
        p = tmp_path / "empty.png"
        p.write_bytes(b"")
        ok, reason = _is_valid_image_path(p)
        assert not ok
        assert "header" in reason.lower()

    def test_truncated_png(self, tmp_path):
        p = tmp_path / "truncated.png"
        p.write_bytes(b"\x89")  # only partial PNG header
        ok, reason = _is_valid_image_path(p)
        assert not ok

    def test_permission_error(self, tmp_path):
        import os
        p = tmp_path / "restricted.png"
        p.write_bytes(b"\x89PNG\r\n\x1a\n")
        p.chmod(0o000)
        try:
            ok, reason = _is_valid_image_path(p)
            assert not ok
            assert "cannot read" in reason
        finally:
            p.chmod(0o644)  # restore for cleanup
        assert p.exists()

    def test_case_insensitive_extension(self, tmp_path):
        for ext in [".PNG", ".Png", ".JPG", ".JPEG", ".GIF", ".TIFF", ".WEBP", ".BMP"]:
            p = tmp_path / f"file{ext}"
            lo = ext.lower()
            if lo == ".png":
                p.write_bytes(b"\x89PNG\r\n\x1a\n")
            elif lo in (".jpg", ".jpeg"):
                p.write_bytes(b"\xff\xd8\xff\xe0")
            elif lo == ".gif":
                p.write_bytes(b"GIF89a\x01\x00\x01\x00")
            elif lo == ".bmp":
                p.write_bytes(b"BM")
            elif lo == ".tiff":
                p.write_bytes(b"II\x2A\x00")
            elif lo == ".webp":
                p.write_bytes(b"RIFF\x00\x00\x00\x00WEBP")
            ok, _ = _is_valid_image_path(p)
            assert ok, f"Failed for extension {ext}"
