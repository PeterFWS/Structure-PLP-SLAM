# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/block.h
        ${CMAKE_CURRENT_SOURCE_DIR}/graph.h
        ${CMAKE_CURRENT_SOURCE_DIR}/energy.h
        ${CMAKE_CURRENT_SOURCE_DIR}/GCoptimization.h
        ${CMAKE_CURRENT_SOURCE_DIR}/LinkedBlockList.h

        ${CMAKE_CURRENT_SOURCE_DIR}/graph.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/GCoptimization.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LinkedBlockList.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/maxflow.cpp
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/solve/GCRANSAC/pearl)
