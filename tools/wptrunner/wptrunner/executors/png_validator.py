from __future__ import annotations

import enum
import math
import struct
import zlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


class ColorSpace(enum.Enum):
    """Colour spaces matching the `colorSpace` BiDi parameter."""
    UNKNOWN = "unknown"
    SRGB = "srgb"
    DISPLAY_P3 = "display-p3"
    REC2020 = "rec2020"
    REC2100_PQ = "rec2100-pq"
    REC2100_HLG = "rec2100-hlg"


class MalformedCaptureError(Exception):
    """PNG payload violates the capture contract."""


class InvalidPNGError(Exception):
    """Bytestream is not a valid PNG."""


@dataclass
class PngInfo:
    """Parsed PNG metadata for contract validation and comparison."""
    width: int
    height: int
    bit_depth: int
    color_type: int
    color_space: ColorSpace
    has_srgb: bool = False
    has_iccp: bool = False
    has_cicp: bool = False
    iccp_profile_name: Optional[str] = None
    cicp_primaries: Optional[int] = None
    cicp_transfer_function: Optional[int] = None
    interlaced: bool = False
    alpha: bool = False


@dataclass
class CaptureContract:
    """Expected PNG properties for a given requested colour space."""
    color_space: ColorSpace
    min_bit_depth: int = 8
    max_bit_depth: int = 16
    require_alpha: bool = False
    allow_alpha: bool = True


_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

_CT_GRAY = 0
_CT_RGB = 2
_CT_INDEXED = 3
_CT_GRAY_ALPHA = 4
_CT_RGBA = 6

_MAX_ICC_PROFILE_SIZE = 4 * 1024 * 1024
_MAX_ICC_TAG_COUNT = 4096
_ICC_TAG_TABLE_OFFSET = 128
_ICC_TAG_RECORD_SIZE = 12
_XYZ_TOLERANCE = 0.002
_TRC_TOLERANCE = 0.002
_D50_ILLUMINANT = b"\x00\x00\xf6\xd6\x00\x01\x00\x00\x00\x00\xd3\x2d"

# ICC RGB colorants use the D50 profile connection space. These are the
# D50-adapted matrices used by Skia for sRGB and Display-P3 profiles.
_SRGB_COLORANTS = (
    (0.436066, 0.222488, 0.013916),
    (0.385147, 0.716873, 0.097076),
    (0.143066, 0.060608, 0.714096),
)
_DISPLAY_P3_COLORANTS = (
    (0.515102, 0.241182, -0.001049),
    (0.291965, 0.692236, 0.041882),
    (0.157153, 0.066582, 0.784378),
)
_ICC_FORWARD_TRANSFORM_TAGS = {
    b"A2B0", b"A2B1", b"A2B2",
    b"D2B0", b"D2B1", b"D2B2", b"D2B3",
}


def _chunk_crc(chunk_type: bytes, data: bytes) -> int:
    return zlib.crc32(chunk_type + data) & 0xFFFFFFFF


def _parse_ihdr(data: bytes) -> Tuple[int, int, int, int, int]:
    width, height, bit_depth, color_type, _compression, _filter_m, interlace = \
        struct.unpack(">LLBBBBB", data[:13])
    return width, height, bit_depth, color_type, interlace


def _parse_cicp(data: bytes) -> Tuple[int, int]:
    if len(data) != 4:
        raise InvalidPNGError("cICP chunk must contain exactly four bytes")
    if data[2] != 0:
        raise InvalidPNGError("cICP matrix coefficients must be zero for RGB")
    if data[3] not in (0, 1):
        raise InvalidPNGError("Invalid cICP video full range flag")
    return data[0], data[1]


def _parse_icc_tags(profile: bytes) -> Dict[bytes, memoryview]:
    if len(profile) < _ICC_TAG_TABLE_OFFSET + 4:
        raise InvalidPNGError("ICC profile is too short")

    declared_size = struct.unpack_from(">I", profile)[0]
    if declared_size != len(profile):
        raise InvalidPNGError(
            f"ICC profile size is {len(profile)}, expected {declared_size}"
        )
    if profile[36:40] != b"acsp":
        raise InvalidPNGError("ICC profile has an invalid signature")
    if profile[68:80] != _D50_ILLUMINANT:
        raise InvalidPNGError("ICC profile has an invalid PCS illuminant")

    tag_count = struct.unpack_from(">I", profile, _ICC_TAG_TABLE_OFFSET)[0]
    if tag_count > _MAX_ICC_TAG_COUNT:
        raise InvalidPNGError("ICC profile has too many tags")

    table_end = (_ICC_TAG_TABLE_OFFSET + 4 +
                 tag_count * _ICC_TAG_RECORD_SIZE)
    if table_end > len(profile):
        raise InvalidPNGError("ICC tag table extends past the profile")

    tag_records = {}
    tag_ranges = set()
    for index in range(tag_count):
        record_offset = (_ICC_TAG_TABLE_OFFSET + 4 +
                         index * _ICC_TAG_RECORD_SIZE)
        signature = profile[record_offset:record_offset + 4]
        offset, size = struct.unpack_from(">II", profile, record_offset + 4)

        if signature in tag_records:
            raise InvalidPNGError(f"Duplicate ICC tag {signature!r}")
        if offset % 4 != 0:
            raise InvalidPNGError(f"ICC tag {signature!r} is not aligned")
        if offset < table_end or size < 8 or offset + size > len(profile):
            raise InvalidPNGError(f"ICC tag {signature!r} is out of bounds")

        tag_records[signature] = (offset, size)
        tag_ranges.add((offset, size))

    previous_end = table_end
    for offset, size in sorted(tag_ranges):
        if offset < previous_end:
            raise InvalidPNGError("ICC tag data ranges overlap")
        previous_end = offset + size

    profile_view = memoryview(profile)
    data_by_range = {
        (offset, size): profile_view[offset:offset + size]
        for offset, size in tag_ranges
    }
    tags = {
        signature: data_by_range[tag_range]
        for signature, tag_range in tag_records.items()
    }

    return tags


def _parse_xyz_tag(data: memoryview) -> Tuple[float, float, float]:
    if len(data) != 20 or data[:4] != b"XYZ " or data[4:8] != b"\x00" * 4:
        raise InvalidPNGError("Invalid ICC XYZ tag")
    return tuple(value / 65536.0
                 for value in struct.unpack_from(">iii", data, 8))


def _evaluate_parametric_trc(
    function_type: int,
    parameters: Tuple[float, ...],
    value: float,
) -> float:
    try:
        if function_type == 0:
            (g,) = parameters
            result = math.pow(value, g)
        elif function_type == 1:
            g, a, b = parameters
            result = (math.pow(a * value + b, g)
                      if value >= -b / a else 0)
        elif function_type == 2:
            g, a, b, c = parameters
            result = (math.pow(a * value + b, g) + c
                      if value >= -b / a else c)
        elif function_type == 3:
            g, a, b, c, d = parameters
            result = (math.pow(a * value + b, g)
                      if value >= d else c * value)
        else:
            g, a, b, c, d, e, f = parameters
            result = (math.pow(a * value + b, g) + e
                      if value >= d else c * value + f)
    except (OverflowError, ValueError, ZeroDivisionError):
        return math.nan

    return min(max(result, 0.0), 1.0)


def _curve_sample(data: memoryview, index: int) -> float:
    return struct.unpack_from(">H", data, 12 + index * 2)[0] / 65535.0


def _evaluate_sampled_trc(data: memoryview, count: int, value: float) -> float:
    sample = value * (count - 1)
    lower = int(sample)
    if lower == count - 1:
        return _curve_sample(data, lower)
    fraction = sample - lower
    lower_value = _curve_sample(data, lower)
    upper_value = _curve_sample(data, lower + 1)
    return lower_value + fraction * (upper_value - lower_value)


def _matches_srgb_trc(data: memoryview) -> bool:
    if len(data) < 8 or data[4:8] != b"\x00" * 4:
        raise InvalidPNGError("Invalid ICC transfer curve tag")

    if data[:4] == b"curv":
        if len(data) < 12:
            raise InvalidPNGError("Truncated ICC curve tag")
        count = struct.unpack_from(">I", data, 8)[0]
        required_size = 12 + count * 2
        if required_size != len(data):
            raise InvalidPNGError("Invalid ICC curve tag size")

        if count > 1:
            for index in range(count):
                value = index / (count - 1)
                if abs(_curve_sample(data, index) -
                       _srgb_to_linear(value)) > _TRC_TOLERANCE:
                    return False

        sample_points = [index / 256.0 for index in range(257)]
        sample_points.append(0.04045)
        for value in sample_points:
            if count == 0:
                actual = value
            elif count == 1:
                gamma = struct.unpack_from(">H", data, 12)[0] / 256.0
                try:
                    actual = math.pow(value, gamma)
                except (OverflowError, ValueError):
                    return False
            else:
                actual = _evaluate_sampled_trc(data, count, value)
            if (not math.isfinite(actual) or
                    abs(actual - _srgb_to_linear(value)) > _TRC_TOLERANCE):
                return False
        return True

    if data[:4] == b"para":
        if len(data) < 12:
            raise InvalidPNGError("Truncated ICC parametric curve tag")
        if data[10:12] != b"\x00\x00":
            raise InvalidPNGError("Invalid ICC parametric curve tag")
        function_type = struct.unpack_from(">H", data, 8)[0]
        parameter_counts = (1, 3, 4, 5, 7)
        if function_type >= len(parameter_counts):
            return False
        parameter_count = parameter_counts[function_type]
        required_size = 12 + parameter_count * 4
        if required_size != len(data):
            raise InvalidPNGError("Invalid ICC parametric curve tag size")
        parameters = tuple(
            value / 65536.0
            for value in struct.unpack_from(
                f">{parameter_count}i", data, 12)
        )
        sample_points = [index / 256.0 for index in range(257)]
        sample_points.append(0.04045)

        branch_point = None
        if function_type in (1, 2):
            a, b = parameters[1:3]
            if a == 0:
                return False
            branch_point = -b / a
        elif function_type in (3, 4):
            branch_point = parameters[4]

        if branch_point is not None and 0 <= branch_point <= 1:
            sample_points.append(branch_point)
            if branch_point > 0:
                sample_points.append(
                    math.nextafter(branch_point, -math.inf)
                )

        for value in sample_points:
            actual = _evaluate_parametric_trc(
                function_type, parameters, value
            )
            if (not math.isfinite(actual) or
                    abs(actual - _srgb_to_linear(value)) > _TRC_TOLERANCE):
                return False
        return True

    return False


def _srgb_to_linear(value: float) -> float:
    if value <= 0.04045:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _matches_colorants(
    actual: Tuple[Tuple[float, float, float], ...],
    expected: Tuple[Tuple[float, float, float], ...],
) -> bool:
    return all(
        abs(actual_value - expected_value) <= _XYZ_TOLERANCE
        for actual_row, expected_row in zip(actual, expected)
        for actual_value, expected_value in zip(actual_row, expected_row)
    )


def _identify_icc_color_space(profile: bytes) -> ColorSpace:
    tags = _parse_icc_tags(profile)
    # Matrix/TRC tags describe the effective device-to-PCS transform only for
    # a simple monitor profile without a higher-priority AToB or DToB transform.
    if (profile[12:16] != b"mntr" or
            profile[16:20] != b"RGB " or
            profile[20:24] != b"XYZ " or
            any(signature in tags for signature in
                _ICC_FORWARD_TRANSFORM_TAGS)):
        return ColorSpace.UNKNOWN

    required_tags = (b"rXYZ", b"gXYZ", b"bXYZ",
                     b"rTRC", b"gTRC", b"bTRC")
    if any(signature not in tags for signature in required_tags):
        return ColorSpace.UNKNOWN

    colorants = tuple(_parse_xyz_tag(tags[signature])
                      for signature in required_tags[:3])
    trc_matches = {}
    for signature in required_tags[3:]:
        trc = tags[signature]
        if id(trc) not in trc_matches:
            trc_matches[id(trc)] = _matches_srgb_trc(trc)
        if not trc_matches[id(trc)]:
            return ColorSpace.UNKNOWN

    if _matches_colorants(colorants, _DISPLAY_P3_COLORANTS):
        return ColorSpace.DISPLAY_P3
    if _matches_colorants(colorants, _SRGB_COLORANTS):
        return ColorSpace.SRGB
    return ColorSpace.UNKNOWN


def _parse_iccp(data: bytes) -> Tuple[str, ColorSpace]:
    null_pos = data.find(b"\x00")
    if null_pos < 1 or null_pos > 79:
        raise InvalidPNGError("Invalid iCCP profile name")
    profile_name_bytes = data[:null_pos]
    if (profile_name_bytes[0] == 0x20 or
            profile_name_bytes[-1] == 0x20 or
            b"  " in profile_name_bytes or
            any(value < 0x20 or 0x7f <= value <= 0xa0
                for value in profile_name_bytes)):
        raise InvalidPNGError("Invalid iCCP profile name")
    if null_pos + 2 > len(data):
        raise InvalidPNGError("Truncated iCCP chunk")
    if data[null_pos + 1] != 0:
        raise InvalidPNGError("Unsupported iCCP compression method")

    compressed_profile = data[null_pos + 2:]
    if not compressed_profile:
        raise InvalidPNGError("iCCP chunk has no compressed profile")

    decompressor = zlib.decompressobj()
    try:
        profile = decompressor.decompress(
            compressed_profile, _MAX_ICC_PROFILE_SIZE + 1)
    except zlib.error as error:
        raise InvalidPNGError(f"Invalid compressed ICC profile: {error}")

    if (len(profile) > _MAX_ICC_PROFILE_SIZE or
            decompressor.unconsumed_tail):
        raise InvalidPNGError("ICC profile exceeds the size limit")
    if not decompressor.eof:
        raise InvalidPNGError("Truncated compressed ICC profile")
    if decompressor.unused_data:
        raise InvalidPNGError("Compressed ICC profile has trailing data")

    profile += decompressor.flush()
    if len(profile) > _MAX_ICC_PROFILE_SIZE:
        raise InvalidPNGError("ICC profile exceeds the size limit")

    profile_name = profile_name_bytes.decode("latin-1")
    return profile_name, _identify_icc_color_space(profile)


def _resolve_color_space_from_chunks(
    has_srgb: bool,
    iccp_color_space: Optional[ColorSpace],
    cicp_primaries: Optional[int],
    cicp_transfer: Optional[int],
) -> ColorSpace:
    if cicp_primaries is not None and cicp_transfer is not None:
        # Display-P3 primaries (12) + sRGB transfer (13)
        if cicp_primaries == 12 and cicp_transfer == 13:
            return ColorSpace.DISPLAY_P3
        # Rec.2020 primaries (9) + sRGB/bt709 transfer
        if cicp_primaries == 9 and cicp_transfer in (1, 13, 14, 15):
            return ColorSpace.REC2020
        # Rec.2020 primaries (9) + PQ transfer (16)
        if cicp_primaries == 9 and cicp_transfer == 16:
            return ColorSpace.REC2100_PQ
        # Rec.2020 primaries (9) + HLG transfer (18)
        if cicp_primaries == 9 and cicp_transfer == 18:
            return ColorSpace.REC2100_HLG
        # sRGB primaries (1) + sRGB transfer (13)
        if cicp_primaries == 1 and cicp_transfer == 13:
            return ColorSpace.SRGB
        return ColorSpace.UNKNOWN

    if iccp_color_space is not None:
        return iccp_color_space

    if has_srgb:
        return ColorSpace.SRGB

    return ColorSpace.UNKNOWN


def parse_png(png_bytes: bytes) -> PngInfo:
    if not png_bytes.startswith(_PNG_SIGNATURE):
        raise InvalidPNGError("Data does not start with PNG signature")

    pos = len(_PNG_SIGNATURE)
    ihdr_found = False
    iend_found = False
    width = height = bit_depth = color_type = interlace = 0
    has_srgb = False
    iccp_name: Optional[str] = None
    iccp_color_space: Optional[ColorSpace] = None
    has_iccp = False
    cicp_primaries: Optional[int] = None
    cicp_transfer: Optional[int] = None
    has_cicp = False

    while pos < len(png_bytes) and not iend_found:
        if pos + 8 > len(png_bytes):
            raise InvalidPNGError("Truncated PNG: cannot read chunk header")

        length, chunk_type = struct.unpack(">L4s", png_bytes[pos : pos + 8])
        pos += 8
        chunk_end = pos + length

        if chunk_end + 4 > len(png_bytes):
            raise InvalidPNGError(
                f"Truncated PNG: chunk {chunk_type!r} extends past end of data"
            )

        chunk_data = png_bytes[pos:chunk_end]
        pos = chunk_end
        crc = struct.unpack(">L", png_bytes[pos : pos + 4])[0]
        pos += 4

        expected_crc = _chunk_crc(chunk_type, chunk_data)
        if crc != expected_crc:
            raise InvalidPNGError(f"CRC mismatch in chunk {chunk_type!r}")

        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, interlace = _parse_ihdr(chunk_data)
            ihdr_found = True
        elif chunk_type == b"sRGB":
            has_srgb = True
        elif chunk_type == b"iCCP":
            has_iccp = True
            iccp_name, iccp_color_space = _parse_iccp(chunk_data)
        elif chunk_type == b"cICP":
            has_cicp = True
            cicp_primaries, cicp_transfer = _parse_cicp(chunk_data)
        elif chunk_type == b"IEND":
            iend_found = True

    if not ihdr_found:
        raise InvalidPNGError("No IHDR chunk found")
    if not iend_found:
        raise InvalidPNGError("No IEND chunk found")

    color_space = _resolve_color_space_from_chunks(
        has_srgb, iccp_color_space, cicp_primaries, cicp_transfer
    )

    return PngInfo(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        color_space=color_space,
        has_srgb=has_srgb,
        has_iccp=has_iccp,
        has_cicp=has_cicp,
        iccp_profile_name=iccp_name,
        cicp_primaries=cicp_primaries,
        cicp_transfer_function=cicp_transfer,
        interlaced=interlace != 0,
        alpha=color_type in (_CT_GRAY_ALPHA, _CT_RGBA),
    )


def validate_contract(png_info: PngInfo, contract: CaptureContract) -> None:
    if png_info.color_space != contract.color_space:
        raise MalformedCaptureError(
            f"Colour space mismatch: PNG reports {png_info.color_space.value}, "
            f"expected {contract.color_space.value}"
        )

    if not (contract.min_bit_depth <= png_info.bit_depth <= contract.max_bit_depth):
        raise MalformedCaptureError(
            f"Bit depth {png_info.bit_depth} is not in accepted range "
            f"[{contract.min_bit_depth}, {contract.max_bit_depth}]"
        )

    if png_info.color_type not in (_CT_RGB, _CT_RGBA):
        raise MalformedCaptureError(
            f"Unexpected PNG colour type {png_info.color_type} (expected RGB or RGBA)"
        )

    if png_info.interlaced:
        raise MalformedCaptureError("Interlaced PNG is not supported")
    if png_info.color_type == _CT_RGBA and not contract.allow_alpha:
        raise MalformedCaptureError("Alpha channel present but not allowed by contract")
    if png_info.color_type == _CT_RGB and contract.require_alpha:
        raise MalformedCaptureError("Alpha channel required but not present")


def decode_pixels_stdlib(
    png_bytes: bytes, info: PngInfo
) -> Tuple[List[List[int]], int, int, int]:
    """Returns (rows, width, height, channels) where *rows* is a list of rows,
    each row is a flat list of channel values (R, G, B[, A])."""
    pos = len(_PNG_SIGNATURE)
    idat_chunks: List[bytes] = []

    while pos < len(png_bytes):
        length, chunk_type = struct.unpack(">L4s", png_bytes[pos : pos + 8])
        pos += 8
        chunk_data = png_bytes[pos : pos + length]
        pos += length + 4  # skip CRC

        if chunk_type == b"IDAT":
            idat_chunks.append(chunk_data)
        elif chunk_type == b"IEND":
            break

    raw = zlib.decompress(b"".join(idat_chunks))

    _CT_TO_CHANNELS = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    channels = _CT_TO_CHANNELS.get(info.color_type, 3)
    bytes_per_channel = info.bit_depth // 8
    pixel_bytes = channels * bytes_per_channel
    stride = info.width * pixel_bytes + 1  # +1 for filter byte

    rows: List[List[int]] = []
    prev_recon: Optional[List[int]] = None

    for y in range(info.height):
        row_start = y * stride
        filter_byte = raw[row_start]
        row_data = list(raw[row_start + 1 : row_start + stride])

        if filter_byte == 0:  # None
            pass
        elif filter_byte == 1:  # Sub
            for i in range(pixel_bytes, len(row_data)):
                row_data[i] = (row_data[i] + row_data[i - pixel_bytes]) & 0xFF
        elif filter_byte == 2:  # Up
            for i in range(len(row_data)):
                up = prev_recon[i] if prev_recon is not None else 0
                row_data[i] = (row_data[i] + up) & 0xFF
        elif filter_byte == 3:  # Average
            for i in range(len(row_data)):
                left = row_data[i - pixel_bytes] if i >= pixel_bytes else 0
                up = prev_recon[i] if prev_recon is not None else 0
                row_data[i] = (row_data[i] + (left + up) // 2) & 0xFF
        elif filter_byte == 4:  # Paeth
            for i in range(len(row_data)):
                left = row_data[i - pixel_bytes] if i >= pixel_bytes else 0
                up = prev_recon[i] if prev_recon is not None else 0
                up_left = (
                    prev_recon[i - pixel_bytes]
                    if prev_recon is not None and i >= pixel_bytes
                    else 0
                )
                row_data[i] = (row_data[i] + _paeth_predictor(left, up, up_left)) & 0xFF
        else:
            raise InvalidPNGError(f"Unknown PNG filter byte: {filter_byte}")

        prev_recon = row_data

        if info.interlaced:
            raise InvalidPNGError("Interlaced PNG is not supported")

        if bytes_per_channel == 1:
            rows.append(list(row_data))
        else:
            row_channels = []
            for i in range(0, len(row_data), 2):
                val = struct.unpack(">H", bytes(row_data[i : i + 2]))[0]
                row_channels.append(val)
            rows.append(row_channels)

    return rows, info.width, info.height, channels


def _paeth_predictor(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


_DEFAULT_CONTRACTS: Dict[str, CaptureContract] = {
    "srgb": CaptureContract(ColorSpace.SRGB, min_bit_depth=8, max_bit_depth=8),
    "display-p3": CaptureContract(ColorSpace.DISPLAY_P3, min_bit_depth=8, max_bit_depth=16),
    "rec2020": CaptureContract(ColorSpace.REC2020, min_bit_depth=8, max_bit_depth=16),
    "rec2100-pq": CaptureContract(ColorSpace.REC2100_PQ, min_bit_depth=10, max_bit_depth=16),
    "rec2100-hlg": CaptureContract(ColorSpace.REC2100_HLG, min_bit_depth=10, max_bit_depth=16),
}


def get_contract(color_space_id: str) -> Optional[CaptureContract]:
    return _DEFAULT_CONTRACTS.get(color_space_id)
