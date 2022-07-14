# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/plane_point_distance_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/plane_point_distance_edge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/point2plane_distance_edge_wrapper.h
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/extended)
