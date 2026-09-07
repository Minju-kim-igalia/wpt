import base64
import io
import logging
import struct
import zlib
from typing import Optional

import pytest

from ..executors.base import RefTestImplementation
from ..executors.png_validator import (
    CaptureContract,
    ColorSpace,
    InvalidPNGError,
    MalformedCaptureError,
    PngInfo,
    decode_pixels_stdlib,
    get_contract,
    parse_png,
    validate_contract,
)


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    c = chunk_type + data
    crc = zlib.crc32(c) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + c + struct.pack(">I", crc)


_PNG_SIG = b"\x89PNG\r\n\x1a\n"

# ICC profile emitted by Chromium's Skia PNG encoder for a Display-P3
# screenshot. The enclosing PNG used "_" as its iCCP profile name.
_CHROMIUM_DISPLAY_P3_ICC = base64.b64decode(
    (
        "AAACCAAAAAAEMAAAbW50clJHQiBYWVogB+AAAQABAAAAAAAAYWNzcAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAEAAPbWAAEAAAAA0y0AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
        "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJZGVzYwAAAPAAAABkclhZWgAAAVQAAAAUZ1hZWgAA"
        "AWgAAAAUYlhZWgAAAXwAAAAUd3RwdAAAAZAAAAAUclRSQwAAAaQAAAAoZ1RSQwAAAaQAAAAo"
        "YlRSQwAAAaQAAAAoY3BydAAAAcwAAAA8bWx1YwAAAAAAAAABAAAADGVuVVMAAABGAAAAHABE"
        "AGkAcwBwAGwAYQB5ACAAUAAzACAARwBhAG0AdQB0ACAAdwBpAHQAaAAgAHMAUgBHAEIAIABU"
        "AHIAYQBuAHMAZgBlAHIAAFhZWiAAAAAAAACD3gAAPb7///+7WFlaIAAAAAAAAEq+AACxNgAA"
        "CrlYWVogAAAAAAAAKDsAABEMAADIzVhZWiAAAAAAAAD21gABAAAAANMtcGFyYQAAAAAABAAA"
        "AAJmZgAA8qcAAA1ZAAAT0AAAClsAAAAAAAAAAG1sdWMAAAAAAAAAAQAAAAxlblVTAAAAIAAA"
        "ABwARwBvAG8AZwBsAGUAIABJAG4AYwAuACAAMgAwADEANg=="
    )
)

_SRGB_COLORANTS = (
    (0.436066, 0.222488, 0.013916),
    (0.385147, 0.716873, 0.097076),
    (0.143066, 0.060608, 0.714096),
)


def _iccp_chunk(profile: bytes, name: bytes = b"_", method: int = 0) -> bytes:
    return _chunk(
        b"iCCP",
        name + b"\x00" + bytes([method]) + zlib.compress(profile),
    )


def _fixed_16_16(value: float) -> bytes:
    return struct.pack(">i", round(value * 65536))


def _xyz_tag(values: tuple[float, float, float]) -> bytes:
    return b"XYZ \x00\x00\x00\x00" + b"".join(
        _fixed_16_16(value) for value in values
    )


def _para_type_4_trc(parameters: tuple[float, ...]) -> bytes:
    assert len(parameters) == 7
    return (
        b"para\x00\x00\x00\x00\x00\x04\x00\x00" +
        b"".join(_fixed_16_16(value) for value in parameters)
    )


def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _sampled_srgb_trc(count: int) -> bytes:
    samples = (
        round(_srgb_to_linear(index / (count - 1)) * 65535)
        for index in range(count)
    )
    return (
        b"curv\x00\x00\x00\x00" + struct.pack(">I", count) +
        b"".join(struct.pack(">H", sample) for sample in samples)
    )


def _make_icc_profile(
    colorants: tuple[tuple[float, float, float], ...],
    trc: bytes,
    extra_tags: tuple[tuple[bytes, bytes], ...] = (),
) -> bytes:
    tags = [
        (b"rXYZ", _xyz_tag(colorants[0])),
        (b"gXYZ", _xyz_tag(colorants[1])),
        (b"bXYZ", _xyz_tag(colorants[2])),
        (b"rTRC", trc),
        (b"gTRC", trc),
        (b"bTRC", trc),
    ]
    tags.extend(extra_tags)
    table_end = 132 + len(tags) * 12
    records = bytearray()
    payload = bytearray()
    data_ranges = {}

    for signature, data in tags:
        if data not in data_ranges:
            offset = table_end + len(payload)
            data_ranges[data] = (offset, len(data))
            payload.extend(data)
            payload.extend(b"\x00" * (-len(payload) % 4))
        records.extend(signature)
        records.extend(struct.pack(">II", *data_ranges[data]))

    profile = bytearray(128)
    profile[8:12] = b"\x04\x30\x00\x00"
    profile[12:16] = b"mntr"
    profile[16:20] = b"RGB "
    profile[20:24] = b"XYZ "
    profile[36:40] = b"acsp"
    profile[68:80] = b"".join(
        _fixed_16_16(value) for value in (0.9642, 1.0, 0.8249)
    )
    profile.extend(struct.pack(">I", len(tags)))
    profile.extend(records)
    profile.extend(payload)
    struct.pack_into(">I", profile, 0, len(profile))
    return bytes(profile)


def _replace_icc_tag(profile: bytes, signature: bytes, data: bytes) -> bytes:
    result = bytearray(profile)
    tag_count = struct.unpack_from(">I", result, 128)[0]
    for index in range(tag_count):
        record_offset = 132 + index * 12
        if result[record_offset:record_offset + 4] != signature:
            continue
        offset, size = struct.unpack_from(">II", result, record_offset + 4)
        assert len(data) == size
        result[offset:offset + size] = data
        return bytes(result)
    raise AssertionError(f"Missing ICC tag {signature!r}")


def _replace_icc_colorants(
    profile: bytes,
    colorants: tuple[tuple[float, float, float], ...],
) -> bytes:
    for signature, values in zip((b"rXYZ", b"gXYZ", b"bXYZ"), colorants):
        profile = _replace_icc_tag(profile, signature, _xyz_tag(values))
    return profile


def _rename_icc_tag(profile: bytes, old: bytes, new: bytes) -> bytes:
    result = bytearray(profile)
    tag_count = struct.unpack_from(">I", result, 128)[0]
    for index in range(tag_count):
        record_offset = 132 + index * 12
        if result[record_offset:record_offset + 4] == old:
            result[record_offset:record_offset + 4] = new
            return bytes(result)
    raise AssertionError(f"Missing ICC tag {old!r}")


def _make_png(
    width: int = 4,
    height: int = 4,
    bit_depth: int = 8,
    color_type: int = 2,  # RGB
    extra_chunks: tuple = (),
    idat: Optional[bytes] = None,
) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0)
    parts = [_PNG_SIG, _chunk(b"IHDR", ihdr)]
    parts.extend(extra_chunks)
    if idat is None:
        channels_map = {0: 1, 2: 3, 4: 2, 6: 4}
        ch_count = channels_map.get(color_type, 3)
        bps = bit_depth // 8
        row = b"\x00" + b"\x00" * width * ch_count * bps
        raw = row * height
        idat = zlib.compress(raw)
    parts.append(_chunk(b"IDAT", idat))
    parts.append(_chunk(b"IEND", b""))
    return b"".join(parts)


class TestParsePng:
    def test_basic_8bit_rgb(self):
        png = _make_png()
        info = parse_png(png)
        assert info.width == 4
        assert info.height == 4
        assert info.bit_depth == 8
        assert info.color_type == 2
        assert info.color_space == ColorSpace.UNKNOWN
        assert not info.alpha

    def test_8bit_rgba(self):
        png = _make_png(color_type=6)
        info = parse_png(png)
        assert info.color_type == 6
        assert info.alpha

    def test_16bit_rgb(self):
        png = _make_png(bit_depth=16)
        info = parse_png(png)
        assert info.bit_depth == 16

    def test_srgb_chunk(self):
        png = _make_png(extra_chunks=(_chunk(b"sRGB", b"\x00"),))
        info = parse_png(png)
        assert info.has_srgb
        assert info.color_space == ColorSpace.SRGB

    def test_cicp_display_p3(self):
        png = _make_png(extra_chunks=(_chunk(b"cICP", bytes([12, 13, 0, 1])),))
        info = parse_png(png)
        assert info.has_cicp
        assert info.cicp_primaries == 12
        assert info.cicp_transfer_function == 13
        assert info.color_space == ColorSpace.DISPLAY_P3

    def test_cicp_rec2100_pq(self):
        png = _make_png(extra_chunks=(_chunk(b"cICP", bytes([9, 16, 0, 1])),))
        info = parse_png(png)
        assert info.color_space == ColorSpace.REC2100_PQ

    def test_cicp_rec2100_hlg(self):
        png = _make_png(extra_chunks=(_chunk(b"cICP", bytes([9, 18, 0, 1])),))
        info = parse_png(png)
        assert info.color_space == ColorSpace.REC2100_HLG

    @pytest.mark.parametrize("chunk_data", [
        bytes([12, 13, 0]),
        bytes([12, 13, 0, 1, 0]),
        bytes([12, 13, 1, 1]),
        bytes([12, 13, 0, 2]),
    ])
    def test_invalid_cicp(self, chunk_data):
        png = _make_png(extra_chunks=(_chunk(b"cICP", chunk_data),))
        with pytest.raises(InvalidPNGError, match="cICP"):
            parse_png(png)

    def test_cicp_takes_precedence_over_iccp_and_srgb(self):
        png = _make_png(extra_chunks=(
            _chunk(b"sRGB", b"\x00"),
            _iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC),
            _chunk(b"cICP", bytes([9, 16, 0, 1])),
        ))
        assert parse_png(png).color_space == ColorSpace.REC2100_PQ

    def test_iccp_takes_precedence_over_srgb(self):
        png = _make_png(extra_chunks=(
            _chunk(b"sRGB", b"\x00"),
            _iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC),
        ))
        assert parse_png(png).color_space == ColorSpace.DISPLAY_P3

    def test_iccp_chromium_display_p3(self):
        png = _make_png(
            extra_chunks=(_iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC),)
        )
        info = parse_png(png)
        assert info.has_iccp
        assert info.iccp_profile_name == "_"
        assert info.color_space == ColorSpace.DISPLAY_P3

    def test_iccp_profile_name_is_not_used_for_display_p3(self):
        png = _make_png(extra_chunks=(
            _iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC, name=b"sRGB"),
        ))
        info = parse_png(png)
        assert info.iccp_profile_name == "sRGB"
        assert info.color_space == ColorSpace.DISPLAY_P3

    def test_iccp_srgb_ignores_display_p3_profile_name(self):
        srgb_profile = _replace_icc_colorants(
            _CHROMIUM_DISPLAY_P3_ICC, _SRGB_COLORANTS
        )
        png = _make_png(extra_chunks=(
            _iccp_chunk(srgb_profile, name=b"Display P3"),
        ))
        info = parse_png(png)
        assert info.iccp_profile_name == "Display P3"
        assert info.color_space == ColorSpace.SRGB

    def test_iccp_sampled_srgb_curve(self):
        profile = _make_icc_profile(
            _SRGB_COLORANTS, _sampled_srgb_trc(257)
        )
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.SRGB

    def test_iccp_non_srgb_curve_is_unknown(self):
        gamma_22 = _para_type_4_trc(
            (2.2, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        )
        profile = _make_icc_profile(
            (
                (0.515102, 0.241182, -0.001049),
                (0.291965, 0.692236, 0.041882),
                (0.157153, 0.066582, 0.784378),
            ),
            gamma_22,
        )
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    def test_iccp_parametric_curve_checks_branch_boundary(self):
        trc = _para_type_4_trc(
            (2.4, 1 / 1.055, 0.055 / 1.055,
             100.0, 0.003, 0.0, 0.0)
        )
        profile = _make_icc_profile(_SRGB_COLORANTS, trc)
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    @pytest.mark.parametrize("mutation", ["reserved", "trailing"])
    def test_iccp_invalid_parametric_curve(self, mutation):
        trc = bytearray(_para_type_4_trc(
            (2.4, 1 / 1.055, 0.055 / 1.055,
             1 / 12.92, 0.04045, 0.0, 0.0)
        ))
        if mutation == "reserved":
            trc[10] = 1
        else:
            trc.extend(b"\x00" * 4)
        profile = _make_icc_profile(_SRGB_COLORANTS, bytes(trc))
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        with pytest.raises(InvalidPNGError, match="parametric curve"):
            parse_png(png)

    def test_iccp_sampled_curve_checks_every_entry(self):
        trc = bytearray(_sampled_srgb_trc(1025))
        struct.pack_into(">H", trc, 12 + 1021 * 2, 0xffff)
        profile = _make_icc_profile(_SRGB_COLORANTS, bytes(trc))
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    def test_iccp_unknown_rgb_profile(self):
        unknown_colorants = list(_SRGB_COLORANTS)
        unknown_colorants[0] = (0.486066, 0.222488, 0.013916)
        profile = _replace_icc_colorants(
            _CHROMIUM_DISPLAY_P3_ICC, tuple(unknown_colorants)
        )
        png = _make_png(extra_chunks=(
            _iccp_chunk(profile, name=b"Display P3"),
        ))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    def test_iccp_missing_required_tag_is_unknown(self):
        profile = _rename_icc_tag(
            _CHROMIUM_DISPLAY_P3_ICC, b"rXYZ", b"xxxx"
        )
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    @pytest.mark.parametrize("offset, value", [
        (12, b"scnr"),
        (16, b"CMYK"),
        (20, b"Lab "),
    ])
    def test_iccp_unsupported_header_is_unknown(self, offset, value):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        profile[offset:offset + 4] = value
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    def test_iccp_forward_transform_takes_precedence(self):
        profile = _make_icc_profile(
            _SRGB_COLORANTS,
            _sampled_srgb_trc(257),
            extra_tags=((b"A2B0", b"mft1\x00\x00\x00\x00"),),
        )
        png = _make_png(extra_chunks=(_iccp_chunk(profile),))
        assert parse_png(png).color_space == ColorSpace.UNKNOWN

    def test_iccp_invalid_compressed_data(self):
        png = _make_png(extra_chunks=(
            _chunk(b"iCCP", b"_\x00\x00not zlib data"),
        ))
        with pytest.raises(InvalidPNGError, match="compressed ICC"):
            parse_png(png)

    def test_iccp_invalid_compression_method(self):
        png = _make_png(extra_chunks=(
            _iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC, method=1),
        ))
        with pytest.raises(InvalidPNGError, match="compression method"):
            parse_png(png)

    def test_iccp_invalid_profile_signature(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        profile[36:40] = b"nope"
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="signature"):
            parse_png(png)

    def test_iccp_invalid_declared_profile_size(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        struct.pack_into(">I", profile, 0, len(profile) - 4)
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="profile size"):
            parse_png(png)

    def test_iccp_out_of_bounds_tag(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        struct.pack_into(">II", profile, 136, len(profile) - 4, 20)
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="out of bounds"):
            parse_png(png)

    def test_iccp_unaligned_tag(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        offset = struct.unpack_from(">I", profile, 136)[0]
        struct.pack_into(">I", profile, 136, offset + 1)
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="not aligned"):
            parse_png(png)

    def test_iccp_overlapping_tags(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        first_offset, first_size = struct.unpack_from(">II", profile, 136)
        struct.pack_into(">II", profile, 148,
                         first_offset + first_size - 4, 20)
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="overlap"):
            parse_png(png)

    def test_iccp_duplicate_tag(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        profile[144:148] = profile[132:136]
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="Duplicate"):
            parse_png(png)

    def test_iccp_invalid_pcs_illuminant(self):
        profile = bytearray(_CHROMIUM_DISPLAY_P3_ICC)
        profile[68:80] = b"\x00" * 12
        png = _make_png(extra_chunks=(_iccp_chunk(bytes(profile)),))
        with pytest.raises(InvalidPNGError, match="PCS illuminant"):
            parse_png(png)

    def test_iccp_trailing_compressed_data(self):
        compressed = zlib.compress(_CHROMIUM_DISPLAY_P3_ICC) + b"trailing"
        png = _make_png(extra_chunks=(
            _chunk(b"iCCP", b"_\x00\x00" + compressed),
        ))
        with pytest.raises(InvalidPNGError, match="trailing data"):
            parse_png(png)

    def test_iccp_profile_size_is_limited(self):
        compressed = zlib.compress(b"\x00" * (4 * 1024 * 1024 + 1))
        png = _make_png(extra_chunks=(
            _chunk(b"iCCP", b"_\x00\x00" + compressed),
        ))
        with pytest.raises(InvalidPNGError, match="size limit"):
            parse_png(png)

    @pytest.mark.parametrize("chunk_data", [
        b"missing separator",
        b"\x00\x00data",
        b"_\x00",
    ])
    def test_iccp_invalid_structure(self, chunk_data):
        png = _make_png(extra_chunks=(_chunk(b"iCCP", chunk_data),))
        with pytest.raises(InvalidPNGError):
            parse_png(png)

    @pytest.mark.parametrize("name", [
        b" leading",
        b"trailing ",
        b"two  spaces",
        b"control\x7f",
        b"x" * 80,
    ])
    def test_iccp_invalid_profile_name(self, name):
        png = _make_png(extra_chunks=(
            _iccp_chunk(_CHROMIUM_DISPLAY_P3_ICC, name=name),
        ))
        with pytest.raises(InvalidPNGError, match="profile name"):
            parse_png(png)

    def test_not_png(self):
        with pytest.raises(InvalidPNGError, match="signature"):
            parse_png(b"not a png file")

    def test_truncated(self):
        with pytest.raises(InvalidPNGError):
            parse_png(_PNG_SIG + b"\x00\x00\x00\x00")

    def test_no_ihdr(self):
        buf = _PNG_SIG + _chunk(b"sRGB", b"\x00") + _chunk(b"IEND", b"")
        with pytest.raises(InvalidPNGError, match="IHDR"):
            parse_png(buf)

    def test_bad_crc(self):
        ihdr = struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0)
        c = b"IHDR" + ihdr
        crc = struct.pack(">I", 0xDEADBEEF)
        buf = _PNG_SIG + struct.pack(">I", len(ihdr)) + c + crc + _chunk(b"IEND", b"")
        with pytest.raises(InvalidPNGError, match="CRC"):
            parse_png(buf)


class TestValidateContract:
    def test_ok_8bit_srgb(self):
        info = PngInfo(4, 4, 8, 2, ColorSpace.SRGB)
        contract = CaptureContract(ColorSpace.SRGB)
        validate_contract(info, contract)

    def test_wrong_color_space(self):
        info = PngInfo(4, 4, 8, 2, ColorSpace.DISPLAY_P3)
        contract = CaptureContract(ColorSpace.SRGB)
        with pytest.raises(MalformedCaptureError, match="Colour space mismatch"):
            validate_contract(info, contract)

    def test_unknown_color_space(self):
        info = PngInfo(4, 4, 8, 2, ColorSpace.UNKNOWN)
        contract = CaptureContract(ColorSpace.SRGB)
        with pytest.raises(
            MalformedCaptureError, match="PNG reports unknown"
        ):
            validate_contract(info, contract)

    def test_bit_depth_too_low(self):
        info = PngInfo(4, 4, 8, 2, ColorSpace.REC2100_PQ)
        contract = CaptureContract(ColorSpace.REC2100_PQ, min_bit_depth=10)
        with pytest.raises(MalformedCaptureError, match="Bit depth"):
            validate_contract(info, contract)

    def test_bit_depth_too_high(self):
        info = PngInfo(4, 4, 16, 2, ColorSpace.SRGB)
        contract = CaptureContract(ColorSpace.SRGB, max_bit_depth=8)
        with pytest.raises(MalformedCaptureError, match="Bit depth"):
            validate_contract(info, contract)

    def test_wrong_color_type(self):
        info = PngInfo(4, 4, 8, 0, ColorSpace.SRGB)  # grayscale
        contract = CaptureContract(ColorSpace.SRGB)
        with pytest.raises(MalformedCaptureError, match="colour type"):
            validate_contract(info, contract)


class TestDecodePixelsStdlib:
    def test_8bit_rgb_roundtrip(self):
        from PIL import Image as PILImage

        for mode, color_type in [("RGB", 2), ("RGBA", 6)]:
            img = PILImage.new(mode, (7, 3))
            for y in range(3):
                for x in range(7):
                    v = (x * 36 + 3, y * 80 + 5, 127)
                    if mode == "RGBA":
                        v = v + (200 - x * 10,)
                    img.putpixel((x, y), v)

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            info = parse_png(png_bytes)
            rows, _w, _h, _ch = decode_pixels_stdlib(png_bytes, info)

            flat = []
            for row in rows:
                flat.extend(row)

            img2 = PILImage.open(io.BytesIO(png_bytes))
            pil_flat = []
            for p in img2.get_flattened_data():
                pil_flat.extend(p)

            assert flat == pil_flat, f"{mode}: stdlib != PIL"

    def test_16bit_grayscale_roundtrip(self):
        from PIL import Image as PILImage

        img = PILImage.new("I;16", (5, 3))
        for y in range(3):
            for x in range(5):
                img.putpixel((x, y), x * 10000 + y * 1000)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        info = parse_png(png_bytes)
        rows, _w, _h, _ch = decode_pixels_stdlib(png_bytes, info)

        flat = []
        for row in rows:
            flat.extend(row)

        img2 = PILImage.open(io.BytesIO(png_bytes))
        pil_flat = list(img2.get_flattened_data())

        assert flat == pil_flat, f"16-bit gray: stdlib={flat} != PIL={pil_flat}"

    def test_16bit_rgb_filter0(self):
        row = b"\x00"
        row += struct.pack(">HHH", 0x0000, 0x0000, 0x0000)
        row += struct.pack(">HHH", 0xFFFF, 0xFFFF, 0xFFFF)
        row += struct.pack(">HHH", 0x8000, 0x4000, 0x2000)
        idat = zlib.compress(row)
        png = _make_png(width=3, height=1, bit_depth=16, idat=idat)
        info = parse_png(png)
        rows, _w, _h, _ch = decode_pixels_stdlib(png, info)
        assert rows[0] == [0, 0, 0, 0xFFFF, 0xFFFF, 0xFFFF, 0x8000, 0x4000, 0x2000]

    def test_16bit_rgba_filter0(self):
        row = b"\x00"
        row += struct.pack(">HHHH", 0x0000, 0xFFFF, 0x8000, 0x4000)
        row += struct.pack(">HHHH", 0x1234, 0x5678, 0x9ABC, 0xDEF0)
        idat = zlib.compress(row)
        png = _make_png(width=2, height=1, bit_depth=16, color_type=6, idat=idat)
        info = parse_png(png)
        rows, _w, _h, _ch = decode_pixels_stdlib(png, info)
        assert rows[0] == [0, 0xFFFF, 0x8000, 0x4000, 0x1234, 0x5678, 0x9ABC, 0xDEF0]


class TestGetContract:
    def test_known(self):
        c = get_contract("display-p3")
        assert c is not None
        assert c.color_space == ColorSpace.DISPLAY_P3

    def test_unknown(self):
        assert get_contract("nonexistent") is None

    def test_srgb_limits_8bit(self):
        c = get_contract("srgb")
        assert c.max_bit_depth == 8

    def test_hdr_requires_more_than_8bit(self):
        for cs in ("rec2100-pq", "rec2100-hlg"):
            c = get_contract(cs)
            assert c is not None
            assert c.min_bit_depth >= 10


class _FakeExecutor:
    """Minimal mock so RefTestImplementation can be instantiated."""
    timeout_multiplier = 1
    subsuite = ""
    screenshot_cache = {}
    reftest_screenshot = "fail"
    logger = logging.getLogger("test")


def _b64(png_bytes: bytes) -> str:
    return base64.b64encode(png_bytes).decode()


@pytest.fixture
def impl():
    return RefTestImplementation(_FakeExecutor())


class TestGetDifferences:
    def test_identical(self, impl):
        png = _make_png(width=3, height=2)
        screenshots = (_b64(png), _b64(png))
        max_diff, count = impl.get_differences(screenshots, urls=["a", "b"])
        assert max_diff == 0
        assert count == 0

    def test_one_pixel_diff(self, impl):
        # Two 2×1 RGB PNGs differing by one channel value.
        row = b"\x00" + b"\xff\x00\x00" + b"\x00\xff\x00"
        png_a = _make_png(width=2, height=1, idat=zlib.compress(row))
        row2 = b"\x00" + b"\xfe\x00\x00" + b"\x00\xff\x00"
        png_b = _make_png(width=2, height=1, idat=zlib.compress(row2))
        screenshots = (_b64(png_a), _b64(png_b))
        max_diff, count = impl.get_differences(screenshots, urls=["a", "b"])
        assert max_diff == 1  # 0xff - 0xfe
        assert count == 1

    def test_size_mismatch(self, impl):
        png_a = _make_png(width=3, height=2)
        png_b = _make_png(width=4, height=2)
        screenshots = (_b64(png_a), _b64(png_b))
        max_diff, count = impl.get_differences(screenshots, urls=["a", "b"])
        assert max_diff == 0
        assert count > 0  # triggers failure

    def test_16bit(self, impl):
        row = b"\x00"
        row += struct.pack(">HHH", 0x0000, 0x0000, 0x0000)
        row += struct.pack(">HHH", 0xFFFF, 0xFFFF, 0xFFFF)
        png_a = _make_png(width=2, height=1, bit_depth=16, idat=zlib.compress(row))
        row2 = b"\x00"
        row2 += struct.pack(">HHH", 0x0000, 0x0000, 0x0000)
        row2 += struct.pack(">HHH", 0xFFFE, 0xFFFF, 0xFFFF)
        png_b = _make_png(width=2, height=1, bit_depth=16, idat=zlib.compress(row2))
        screenshots = (_b64(png_a), _b64(png_b))
        max_diff, count = impl.get_differences(screenshots, urls=["a", "b"])
        assert max_diff == 1
        assert count == 1

    def test_invalid_base64(self, impl):
        max_diff, count = impl.get_differences(("not base64", "also bad"), urls=["a", "b"])
        assert max_diff is None
        assert count is None

    def test_alpha_skipped(self, impl):
        # Two 2×1 RGBA PNGs with same RGB but different alpha.
        row_a = b"\x00" + b"\x80\x40\x20\xff" + b"\x10\x20\x30\x80"
        png_a = _make_png(width=2, height=1, color_type=6, idat=zlib.compress(row_a))
        row_b = b"\x00" + b"\x80\x40\x20\x00" + b"\x10\x20\x30\x00"
        png_b = _make_png(width=2, height=1, color_type=6, idat=zlib.compress(row_b))
        screenshots = (_b64(png_a), _b64(png_b))

        # Without contract: composited against black → alpha matters.
        max_diff, count = impl.get_differences(screenshots, urls=["a", "b"])
        assert max_diff == 128  # 0x80*255//255 vs 0x80*0//255
        assert count == 2  # both pixels differ

        # With contract: alpha dropped, raw RGB compared → identical.
        from ..executors.png_validator import CaptureContract, ColorSpace
        contract = CaptureContract(ColorSpace.SRGB)
        max_diff2, count2 = impl.get_differences(
            screenshots, urls=["a", "b"], contract=contract
        )
        assert max_diff2 == 0
        assert count2 == 0

    def test_solid_colour_warns(self, impl):
        row = b"\x00" + b"\xFF\x00\x00" * 2
        png = _make_png(width=2, height=2, idat=zlib.compress(row * 2))
        impl.message = []
        impl.get_differences((_b64(png), _b64(png)), urls=["a", "b"])
        assert any("solid colour" in m for m in impl.message)
        assert any("FF0000" in m for m in impl.message)


class TestCheckPass:
    def _hash(self, screenshots):
        h = ["a", "b"]
        return (h[:len(screenshots[0])], h[:len(screenshots[1])])

    def test_pass_identical(self, impl):
        png = _make_png()
        b64 = _b64(png)
        screenshots = ([b64], [b64])
        hashes = (["h1"], ["h1"])
        result, page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", None
        )
        assert result is True
        assert page == -1

    def test_fail_different(self, impl):
        png_a = _make_png()
        # Make a different PNG by changing a pixel.
        row = b"\x00" + b"\x80\x80\x80" * 4
        png_b = _make_png(width=4, height=1, idat=zlib.compress(row))
        screenshots = ([_b64(png_a)], [_b64(png_b)])
        hashes = (["h1"], ["h2"])
        result, _page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", None
        )
        assert result is False

    def test_contract_valid(self, impl):
        png = _make_png(extra_chunks=(_chunk(b"cICP", bytes([12, 13, 0, 1])),))
        b64 = _b64(png)
        screenshots = ([b64], [b64])
        hashes = (["h1"], ["h1"])
        result, page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", None,
            color_space="display-p3",
        )
        assert result is True
        assert page == -1

    def test_contract_violation_wrong_space(self, impl):
        # PNG is sRGB but contract expects Display-P3.
        png = _make_png()
        b64 = _b64(png)
        screenshots = ([b64], [b64])
        hashes = (["h1"], ["h1"])
        result, _page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", None,
            color_space="display-p3",
        )
        # Contract violation → (None, page_idx)
        assert result is None

    def test_no_contract_no_validation(self, impl):
        # Without color_space, even a Display-P3 PNG is compared without error.
        png = _make_png(extra_chunks=(_chunk(b"cICP", bytes([12, 13, 0, 1])),))
        b64 = _b64(png)
        screenshots = ([b64], [b64])
        hashes = (["h1"], ["h1"])
        result, page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", None
        )
        assert result is True
        assert page == -1


    def test_fuzzy_with_high_bit_depth_fails(self, impl):
        png = _make_png(bit_depth=16, extra_chunks=(_chunk(b"cICP", bytes([12, 13, 0, 1])),))
        b64 = _b64(png)
        screenshots = ([b64], [b64])
        hashes = (["h1"], ["h1"])
        result, page = impl.check_pass(
            hashes, screenshots, ["test", "ref"], "==", ([10, 10], [100, 100]),
            color_space="display-p3",
        )
        assert result is None
