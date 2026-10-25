idf_component_register(
    SRCS
        "main.c"
        "espdl_model.c"
        "i2c_lcd.c"
    INCLUDE_DIRS
        "."
    REQUIRES
        driver
        esp_timer
        freertos
)

set_source_files_properties(espdl_model.c PROPERTIES COMPILE_FLAGS "-O3 -funroll-loops")
