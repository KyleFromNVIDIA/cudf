/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cudf/strings/string_view.cuh>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/stream>

namespace cudf::strings::detail {

/**
 * @brief Populate preallocated storage with the input column's string views.
 *
 * @param input Input strings column
 * @param output Output span with the same size as input
 * @param stream Stream used for device operations
 */
void populate_string_views(strings_column_view input,
                           device_span<string_view> output,
                           cuda::stream_ref stream);

}  // namespace cudf::strings::detail
