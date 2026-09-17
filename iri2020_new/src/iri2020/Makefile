# Convenience wrapper around the CMake build in build/.
#
# CMake bakes absolute paths (source dir, compiler, generator internals)
# into build/CMakeCache.txt and build/Makefile at configure time. Those
# files are auto-generated -- hand-editing them is a losing game, since the
# next real reconfigure silently overwrites any manual fix. Once this
# folder is copied, cloned, or moved to a new location (or a different
# machine/user), the cached paths go stale and plain `make -C build` starts
# failing in confusing ways.
#
# This wrapper compares the cached source directory against where this
# Makefile actually lives right now, and transparently wipes + reconfigures
# the build tree whenever they differ (or it doesn't exist yet) -- so
# `make` always finds the right folders on its own, with no manual path
# editing ever required.
#
# Usage: run `make` from this directory (or `make -C <this dir>` from
# anywhere) to build both executables. For anything the generated build
# needs beyond that (e.g. `ctest`, a specific object target), run it
# directly in build/ once `make` here has ensured it's correctly configured.

SRC_DIR   := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
BUILD_DIR := $(SRC_DIR)/build
CACHE     := $(BUILD_DIR)/CMakeCache.txt

.PHONY: all check-configured reconfigure clean distclean

all: check-configured
	$(MAKE) -C $(BUILD_DIR)

check-configured:
	@if [ ! -f '$(CACHE)' ]; then \
		echo "iri2020: no build/ configured yet -- running cmake."; \
		cmake -S '$(SRC_DIR)' -B '$(BUILD_DIR)'; \
	else \
		cached_src="$$(sed -n 's/^iri2020_SOURCE_DIR:STATIC=//p' '$(CACHE)')"; \
		if [ "$$cached_src" != "$(SRC_DIR)" ]; then \
			echo "iri2020: build/ was configured for '$$cached_src',"; \
			echo "iri2020: but this source tree is now at '$(SRC_DIR)' -- reconfiguring."; \
			rm -rf '$(BUILD_DIR)'; \
			cmake -S '$(SRC_DIR)' -B '$(BUILD_DIR)'; \
		fi; \
	fi

# Force a from-scratch reconfigure regardless of whether the cache looks current.
reconfigure:
	rm -rf $(BUILD_DIR)
	cmake -S $(SRC_DIR) -B $(BUILD_DIR)

clean: check-configured
	$(MAKE) -C $(BUILD_DIR) clean

distclean:
	rm -rf $(BUILD_DIR)
