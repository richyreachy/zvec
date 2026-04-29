include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 REQUIRED)

set(SRC_LISTS
        binding.cc
        model/python_collection.cc
        model/python_doc.cc
        model/param/python_param.cc
        model/schema/python_schema.cc
        model/common/python_config.cc
        model/common/python_plugin.cc
        typing/python_type.cc
)

pybind11_add_module(_zvec ${SRC_LISTS})

if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
        target_link_libraries(_zvec PRIVATE
                -Wl,--whole-archive
                $<TARGET_FILE:core_knn_flat_static>
                $<TARGET_FILE:core_knn_flat_sparse_static>
                $<TARGET_FILE:core_knn_hnsw_static>
                $<TARGET_FILE:core_knn_hnsw_rabitq_static>
                $<TARGET_FILE:core_knn_hnsw_sparse_static>
                $<TARGET_FILE:core_knn_ivf_static>
                $<TARGET_FILE:core_knn_cluster_static>
                $<TARGET_FILE:core_mix_reducer_static>
                $<TARGET_FILE:core_metric_static>
                $<TARGET_FILE:core_utility_static>
                $<TARGET_FILE:core_quantizer_static>
                $<TARGET_FILE:core_plugin>
                -Wl,--no-whole-archive
                zvec_db
                ${CMAKE_DL_LIBS}
        )
        target_link_options(_zvec PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
        )
    else ()
        target_link_libraries(_zvec PRIVATE
                -Wl,--whole-archive
                $<TARGET_FILE:core_knn_flat_static>
                $<TARGET_FILE:core_knn_flat_sparse_static>
                $<TARGET_FILE:core_knn_hnsw_static>
                $<TARGET_FILE:core_knn_hnsw_rabitq_static>
                $<TARGET_FILE:core_knn_hnsw_sparse_static>
                $<TARGET_FILE:core_knn_ivf_static>
                $<TARGET_FILE:core_knn_cluster_static>
                $<TARGET_FILE:core_mix_reducer_static>
                $<TARGET_FILE:core_metric_static>
                $<TARGET_FILE:core_utility_static>
                $<TARGET_FILE:core_quantizer_static>
                $<TARGET_FILE:core_plugin>
                -Wl,--no-whole-archive
                zvec_db
                ${CMAKE_DL_LIBS}
        )
        # DiskAnn is built as an optional runtime plugin (libzvec_diskann_plugin.so).
        # Users load it explicitly via zvec::LoadDiskAnnPlugin() after checking
        # libaio availability; so it is NOT linked statically here and libaio is
        # no longer a build-time dependency of the python extension.
        # Ensure the plugin .so next to the extension is discoverable.
        set_target_properties(_zvec PROPERTIES
                BUILD_RPATH "$ORIGIN"
                INSTALL_RPATH "$ORIGIN"
        )
        target_link_options(_zvec PRIVATE
                "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map"
        )
    endif()
elseif (APPLE)
    target_link_libraries(_zvec PRIVATE
            -Wl,-force_load,$<TARGET_FILE:core_knn_flat_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_flat_sparse_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_rabitq_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_hnsw_sparse_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_ivf_static>
            -Wl,-force_load,$<TARGET_FILE:core_knn_cluster_static>
            -Wl,-force_load,$<TARGET_FILE:core_mix_reducer_static>
            -Wl,-force_load,$<TARGET_FILE:core_metric_static>
            -Wl,-force_load,$<TARGET_FILE:core_utility_static>
            -Wl,-force_load,$<TARGET_FILE:core_quantizer_static>
            -Wl,-force_load,$<TARGET_FILE:core_plugin>
            zvec_db
    )
    target_link_libraries(_zvec PRIVATE
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/exports.mac
    )
elseif (MSVC)
    set(_zvec_whole_archive_libs
            core_knn_flat_static
            core_knn_flat_sparse_static
            core_knn_hnsw_static
            core_knn_hnsw_sparse_static
            core_knn_ivf_static
            core_knn_cluster_static
            core_mix_reducer_static
            core_metric_static
            core_utility_static
            core_quantizer_static
            core_plugin
    )
    target_link_libraries(_zvec PRIVATE
            ${_zvec_whole_archive_libs}
            zvec_db
    )
    foreach(_lib ${_zvec_whole_archive_libs})
        target_link_options(_zvec PRIVATE
                "/WHOLEARCHIVE:$<TARGET_FILE:${_lib}>"
        )
    endforeach()
endif ()

target_include_directories(_zvec PRIVATE ${PYBIND11_INCLUDE_DIR} ${PROJECT_ROOT_DIR}/src ${PROJECT_ROOT_DIR}/src/binding/python/include)