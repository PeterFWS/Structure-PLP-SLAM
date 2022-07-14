target_sources(${PROJECT_NAME}
        PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/types_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/descriptor_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/line_descriptor_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/precomp_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_descriptor_custom.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/binary_descriptor_matcher.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bitarray_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/bitops_custom.hpp
        ${CMAKE_CURRENT_SOURCE_DIR}/draw_custom.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/LSDDetector_custom.cpp
)

# Install headers
file(GLOB HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp")
install(FILES ${HEADERS}
        DESTINATION ${PLPSLAM_INCLUDE_INSTALL_DIR}/feature/line_descriptor)