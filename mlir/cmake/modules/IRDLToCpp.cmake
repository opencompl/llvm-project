function(add_irdl_to_cpp_target target)
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/{target}.cpp.inc
    COMMAND $<TARGET_FILE:mlir-irdl-to-cpp> {target}.cpp.inc
    DEPENDS mlir-irdl-to-cpp
    COMMENT "Building ${target}..."
  )
endfunction()
