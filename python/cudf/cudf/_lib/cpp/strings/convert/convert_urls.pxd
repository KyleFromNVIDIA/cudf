# Copyright (c) 2020-2021, NVIDIA CORPORATION.

from libcpp.memory cimport unique_ptr

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.column.column_view cimport column_view


cdef extern from "cudf/strings/convert/convert_urls.hpp" namespace \
        "cudf::strings" nogil:
    cdef unique_ptr[column] url_encode(
        column_view input_col) except +

    cdef unique_ptr[column] url_decode(
        column_view input_col) except +
