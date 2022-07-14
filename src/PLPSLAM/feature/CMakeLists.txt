# Add sources
target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_params.h
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor_node.h
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_params.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/orb_extractor_node.cc

        ${CMAKE_CURRENT_SOURCE_DIR}/line_extractor.h
        ${CMAKE_CURRENT_SOURCE_DIR}/line_extractor.cc
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.h")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/feature)

# FW: Append subdirectory for customized LSD and LBD (line segment detector and descriptor)
add_subdirectory(line_descriptor)