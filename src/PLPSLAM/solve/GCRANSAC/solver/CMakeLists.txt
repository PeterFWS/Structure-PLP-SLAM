# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE

        # ${CMAKE_CURRENT_SOURCE_DIR}/solver_dls_pnp.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_engine.h

        # ${CMAKE_CURRENT_SOURCE_DIR}/solver_epnp_lm.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_essential_matrix_five_point_stewenius.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_fundamental_matrix_eight_point.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_fundamental_matrix_plane_and_parallax.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_fundamental_matrix_seven_point.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_homography_four_point.h
        ${CMAKE_CURRENT_SOURCE_DIR}/solver_linear_model.h

        # ${CMAKE_CURRENT_SOURCE_DIR}/solver_p3p.h
        # ${CMAKE_CURRENT_SOURCE_DIR}/solver_rigid_transformation_svd.h
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/solve/GCRANSAC/solver)