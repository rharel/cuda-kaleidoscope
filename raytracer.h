#pragma once

#include "config.h"
#include "cudu.h"

namespace kaleido
{
    cudu::host::Array3D<unsigned char> raytraced_image(const Config& config);
}
