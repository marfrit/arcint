# patches/

A browsing mirror of the plugin patch series 0003–0067 (`marfrit-openvino
+p20`), plus the two patches that were measured and deliberately never
applied: 0001 (an instrument that forces a null implementation, so a node's
cost can be measured by removal) and 0002 (a raised horizontal FC fusion
bound: wrong output on the MoE quartet, no gain on the GDN sets).

The series the runtime is built with lives in
`contrib/packaging/marfrit-openvino/patches/` — that directory is what the
recipe applies against the pinned OpenVINO commit, and its `README.md` is
the per-patch record (what each changes and its standing measured effect).
This directory mirrors it file for file so the series can be read at the top
of the tree; a device-free test (`tests/test_patches_mirror.cpp`) fails when
a recipe patch is missing here or differs by a byte.
