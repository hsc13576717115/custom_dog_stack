# custom_dog_hardware

This package owns the USB-to-four-RS485 transport boundary. It currently
defines the stable motor command/state interface only. Protocol frames,
serial scheduling and hardware tests are intentionally not marked complete.

The implementation must provide command timeout, CRC/status validation,
temperature limits and a no-allocation real-time exchange path before use.
