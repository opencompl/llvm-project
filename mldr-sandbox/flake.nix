{
  description = "A devShell example";

  inputs = {
    nixpkgs.url      = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url  = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        gccForLibs = pkgs.stdenv.cc.cc;
      in
      with pkgs;
      {

        devShell = mkShell rec {
          APPEND_LIBRARY_PATH = lib.makeLibraryPath [
            stdenv.cc.cc.lib
            ];

          LD_LIBRARY_PATH = APPEND_LIBRARY_PATH;
          PKG_CONFIG_PATH = APPEND_LIBRARY_PATH;
          
          NIX_LDFLAGS="-L${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version}";

          CFLAGS="-B${gccForLibs}/lib/gcc/${targetPlatform.config}/${gccForLibs.version} -B ${stdenv.cc.libc}/lib";

          cmakeFlags = [
            "-DC_INCLUDE_DIRS=${stdenv.cc.libc.dev}/include"
            "-GNinja"
            # Debug for debug builds
            "-DCMAKE_BUILD_TYPE=Release"
            # inst will be our installation prefix
            "-DCMAKE_INSTALL_PREFIX=../inst"
            "-DLLVM_INSTALL_TOOLCHAIN_ONLY=ON"
            # change this to enable the projects you need
            "-DLLVM_ENABLE_PROJECTS=clang;mlir"
            # enable libcxx* to come into play at runtimes
            "-DLLVM_ENABLE_RUNTIMES=libcxx;libcxxabi"
            # this makes llvm only to produce code for the current platform, this saves CPU time, change it to what you need
            "-DLLVM_TARGETS_TO_BUILD=host"
          ];

          packages = [
            cmake
            gcc
            libgcc
            ninja
            mold
          ];

          nativeBuildInputs = [
            pkg-config
          ];
        };
      }
    );
  }