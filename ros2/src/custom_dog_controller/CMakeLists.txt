cmake_minimum_required(VERSION 3.8)
project(custom_dog_controller)

find_package(ament_cmake REQUIRED)

add_library(${PROJECT_NAME} INTERFACE)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_include_directories(
  ${PROJECT_NAME}
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

if(BUILD_TESTING)
  enable_testing()
  add_executable(deployment_state_machine_test test/deployment_state_machine_test.cpp)
  target_link_libraries(deployment_state_machine_test PRIVATE ${PROJECT_NAME})
  add_test(NAME deployment_state_machine_test COMMAND deployment_state_machine_test)
  add_executable(history_observation_test test/history_observation_test.cpp)
  target_link_libraries(history_observation_test PRIVATE ${PROJECT_NAME})
  add_test(NAME history_observation_test COMMAND history_observation_test)
endif()

install(DIRECTORY include/ DESTINATION include)
install(
  TARGETS ${PROJECT_NAME}
  EXPORT export_${PROJECT_NAME}
)

ament_export_targets(export_${PROJECT_NAME})
ament_export_include_directories(include)
ament_package()
