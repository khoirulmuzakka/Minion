if(NOT TARGET minion)
    message(FATAL_ERROR "tests/ must be added from the top-level Minion build.")
endif()

add_executable(minion_test test_minion.cpp)
target_link_libraries(minion_test PRIVATE minion)

if(TARGET minion_cec)
    target_link_libraries(minion_test PRIVATE minion_cec)
else()
    message(FATAL_ERROR "minion_test requires MINION_BUILD_CEC=ON.")
endif()

if(WIN32)
    add_custom_command(TARGET minion_test POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:minion_test>
            $<TARGET_FILE_DIR:minion_test>
        COMMAND_EXPAND_LISTS
    )
endif()

if(MSVC)
    set_target_properties(minion_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/$<0:>
    )
else()
    set_target_properties(minion_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

add_test(NAME minion_integration COMMAND minion_test)
