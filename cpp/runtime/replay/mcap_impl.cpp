// MCAP is header-only: exactly one translation unit in the program must define
// MCAP_IMPLEMENTATION to emit the out-of-line definitions. This file exists for
// that and nothing else. Defining it in a header would produce duplicate
// symbols the moment two TUs included it.
#define MCAP_IMPLEMENTATION
#include <mcap/reader.hpp>
#include <mcap/writer.hpp>
