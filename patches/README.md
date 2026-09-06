# patches/

A browsing mirror of the plugin patch series, plus the two patches that were
measured and deliberately never applied (0001, 0002).

The series the runtime is built with lives in
`contrib/packaging/marfrit-openvino/patches/` — that directory is what the
recipe applies against the pinned OpenVINO commit, and its `README.md` is
the per-patch record (what each changes, what was measured, upstream
status). This directory mirrors it file for file so the series can be read
at the top of the tree; a device-free test (`tests/test_patches_mirror.cpp`)
fails when the two drift, which is how this mirror fell five patches behind
between 0.3.0 and 0.4.1 without anyone noticing.
