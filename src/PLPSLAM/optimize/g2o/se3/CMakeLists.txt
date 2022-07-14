# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex.h
        ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container.h
        ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/perspective_reproj_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_pose_opt_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_reproj_edge.h
        ${CMAKE_CURRENT_SOURCE_DIR}/reproj_edge_line3d_orthonormal.h
        ${CMAKE_CURRENT_SOURCE_DIR}/pose_opt_edge_line3d_orthonormal.h
        ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/shot_vertex_container.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/perspective_pose_opt_edge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/perspective_reproj_edge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_pose_opt_edge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/equirectangular_reproj_edge.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/reproj_edge_line3d_orthonormal.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/pose_opt_edge_line3d_orthonormal.cc
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/optimize/g2o/se3)
