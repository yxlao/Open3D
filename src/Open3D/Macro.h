#pragma once

#define OPEN3D_CONCATENATE_IMPL(s1, s2) s1##s2
#define OPEN3D_CONCATENATE(s1, s2) OPEN3D_CONCATENATE_IMPL(s1, s2)
