#!/usr/bin/env bash
# Sets IRI2020_PATH to the location of the compiled IRI2020 Fortran driver,
# consumed by EDPSamples.get_IRI2020_EDP (see EDPSamples/edp_samples.py).
#
# The path is derived from where THIS script itself lives on disk, not
# hardcoded, so it keeps working if the IonosphereTomography repository is
# cloned, copied, or moved elsewhere -- the same self-locating approach used
# by iri2020_new/src/iri2020/Makefile for the CMake build itself.
#
# This must be SOURCED, not executed, so the exported variable reaches your
# current shell rather than a throwaway subshell:
#
#     source init_iri2020_env.sh
#   or
#     . init_iri2020_env.sh
#
# Works in both bash and zsh.

if [ -n "${ZSH_VERSION:-}" ]; then
    _iri2020_env_script="${(%):-%N}"
else
    _iri2020_env_script="${BASH_SOURCE[0]:-$0}"
fi
_iri2020_env_dir="$(cd "$(dirname "${_iri2020_env_script}")" && pwd)"

export IRI2020_PATH="${_iri2020_env_dir}/iri2020_new/src/iri2020"

if [ -x "${IRI2020_PATH}/iri2020_namelist_driver" ]; then
    echo "IRI2020_PATH set to: ${IRI2020_PATH}"
else
    echo "IRI2020_PATH set to: ${IRI2020_PATH}" >&2
    echo "Warning: no compiled iri2020_namelist_driver found there yet." >&2
    echo "Run 'make' in ${IRI2020_PATH} to build it." >&2
fi

unset _iri2020_env_script _iri2020_env_dir
