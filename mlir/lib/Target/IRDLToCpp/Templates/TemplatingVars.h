// this file defines all the mappings of string macros to string getters
// "__DIALECT_NAME__"
// we decay the pointer because i want to know what type it is
  { "__DIALECT_NAME__", +[](const DialectStrings& dialect) -> std::string { return dialect.dialectName; } }
, { "__NAMESPACE_OPEN__", +[](const DialectStrings& dialect) -> std::string  { return dialect.namespaceOpen; } }
, { "__NAMESPACE_CLOSE__", +[](const DialectStrings& dialect)  -> std::string { return dialect.namespaceClose; } }