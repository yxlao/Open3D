The cpp/cuda system is not stable now. The interfaces may change greatly,
therefore it is improper to derive python interfaces at current.

A cpp-based system will first be implemented here in order to reproduce the
result of the original version.
Only after the interfaces are stable shall we start to work on python binds.

The version is simplified, without 5-pt matching inside a fragment -- we assume
the odometry is accurate enough. With such a configuration, pose graph is
not necessary inside a fragment. This may reduce a little bit accuracy.