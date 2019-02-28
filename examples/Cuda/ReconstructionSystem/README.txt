The cpp/cuda system is not stable now. The interfaces may change greatly,
therefore it is improper to derive python interfaces at current.

A cpp-based system will first be implemented here in order to reproduce the
result of the original version.
Only after the interfaces are stable shall we start to work on python binds.

This version only supports Colored ICP and FGR. Point-to-point and
point-to-plane ICP, as well as RANSAC-based global registration will be
supported later.