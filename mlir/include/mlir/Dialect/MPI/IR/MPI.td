//===- MPIBase.td - Base defs for mpi dialect ---------------*- tablegen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MPI_IR_MPI
#define MLIR_DIALECT_MPI_IR_MPI

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/EnumAttr.td"

def MPI_Dialect : Dialect {
  let name = "mpi";
  let cppNamespace = "::mlir::mpi";
  let description = [{
    This dialect models the Message Passing Interface (MPI), version 4.0. It is
    meant to serve as an interfacing dialect that is targeted by higher-level dialects.
    The MPI dialect itself can be lowered to multiple MPI implementations and hide
    differences in ABI. The dialect models the functions of the MPI specification as
    close to 1:1 as possible while preserving SSA value semantics where it makes sense,
    and uses `memref` types instead of bare pointers.

    This dialect is under active development, and while stability is an
    eventual goal, it is not guaranteed at this juncture. Given the early state,
    it is recommended to inquire further prior to using this dialect.

    For an in-depth documentation of the MPI library interface, please refer to official documentation
    such as the [OpenMPI online documentation](https://www.open-mpi.org/doc/current/).
  }];

  let usePropertiesForAttributes = 1;
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
}

//===----------------------------------------------------------------------===//
// Error classes enum:
//===----------------------------------------------------------------------===//



def MpiCodeSuccess : I32EnumAttrCase<"MPI_SUCCESS", 0, "MPI_SUCCESS">;
def MpiCodeErrAccess : I32EnumAttrCase<"MPI_ERR_ACCESS", 1, "MPI_ERR_ACCESS">;
def MpiCodeErrAmode : I32EnumAttrCase<"MPI_ERR_AMODE", 2, "MPI_ERR_AMODE">;
def MpiCodeErrArg : I32EnumAttrCase<"MPI_ERR_ARG", 3, "MPI_ERR_ARG">;
def MpiCodeErrAssert : I32EnumAttrCase<"MPI_ERR_ASSERT", 4, "MPI_ERR_ASSERT">;
def MpiCodeErrBadFile : I32EnumAttrCase<"MPI_ERR_BAD_FILE", 5, "MPI_ERR_BAD_FILE">;
def MpiCodeErrBase : I32EnumAttrCase<"MPI_ERR_BASE", 6, "MPI_ERR_BASE">;
def MpiCodeErrBuffer : I32EnumAttrCase<"MPI_ERR_BUFFER", 7, "MPI_ERR_BUFFER">;
def MpiCodeErrComm : I32EnumAttrCase<"MPI_ERR_COMM", 8, "MPI_ERR_COMM">;
def MpiCodeErrConversion : I32EnumAttrCase<"MPI_ERR_CONVERSION", 9, "MPI_ERR_CONVERSION">;
def MpiCodeErrCount : I32EnumAttrCase<"MPI_ERR_COUNT", 10, "MPI_ERR_COUNT">;
def MpiCodeErrDims : I32EnumAttrCase<"MPI_ERR_DIMS", 11, "MPI_ERR_DIMS">;
def MpiCodeErrDisp : I32EnumAttrCase<"MPI_ERR_DISP", 12, "MPI_ERR_DISP">;
def MpiCodeErrDupDatarep : I32EnumAttrCase<"MPI_ERR_DUP_DATAREP", 13, "MPI_ERR_DUP_DATAREP">;
def MpiCodeErrErrhandler : I32EnumAttrCase<"MPI_ERR_ERRHANDLER", 14, "MPI_ERR_ERRHANDLER">;
def MpiCodeErrFile : I32EnumAttrCase<"MPI_ERR_FILE", 15, "MPI_ERR_FILE">;
def MpiCodeErrFileExists : I32EnumAttrCase<"MPI_ERR_FILE_EXISTS", 16, "MPI_ERR_FILE_EXISTS">;
def MpiCodeErrFileInUse : I32EnumAttrCase<"MPI_ERR_FILE_IN_USE", 17, "MPI_ERR_FILE_IN_USE">;
def MpiCodeErrGroup : I32EnumAttrCase<"MPI_ERR_GROUP", 18, "MPI_ERR_GROUP">;
def MpiCodeErrInfo : I32EnumAttrCase<"MPI_ERR_INFO", 19, "MPI_ERR_INFO">;
def MpiCodeErrInfoKey : I32EnumAttrCase<"MPI_ERR_INFO_KEY", 20, "MPI_ERR_INFO_KEY">;
def MpiCodeErrInfoNokey : I32EnumAttrCase<"MPI_ERR_INFO_NOKEY", 21, "MPI_ERR_INFO_NOKEY">;
def MpiCodeErrInfoValue : I32EnumAttrCase<"MPI_ERR_INFO_VALUE", 22, "MPI_ERR_INFO_VALUE">;
def MpiCodeErrInStatus : I32EnumAttrCase<"MPI_ERR_IN_STATUS", 23, "MPI_ERR_IN_STATUS">;
def MpiCodeErrIntern : I32EnumAttrCase<"MPI_ERR_INTERN", 24, "MPI_ERR_INTERN">;
def MpiCodeErrIo : I32EnumAttrCase<"MPI_ERR_IO", 25, "MPI_ERR_IO">;
def MpiCodeErrKeyval : I32EnumAttrCase<"MPI_ERR_KEYVAL", 26, "MPI_ERR_KEYVAL">;
def MpiCodeErrLocktype : I32EnumAttrCase<"MPI_ERR_LOCKTYPE", 27, "MPI_ERR_LOCKTYPE">;
def MpiCodeErrName : I32EnumAttrCase<"MPI_ERR_NAME", 28, "MPI_ERR_NAME">;
def MpiCodeErrNoMem : I32EnumAttrCase<"MPI_ERR_NO_MEM", 29, "MPI_ERR_NO_MEM">;
def MpiCodeErrNoSpace : I32EnumAttrCase<"MPI_ERR_NO_SPACE", 30, "MPI_ERR_NO_SPACE">;
def MpiCodeErrNoSuchFile : I32EnumAttrCase<"MPI_ERR_NO_SUCH_FILE", 31, "MPI_ERR_NO_SUCH_FILE">;
def MpiCodeErrNotSame : I32EnumAttrCase<"MPI_ERR_NOT_SAME", 32, "MPI_ERR_NOT_SAME">;
def MpiCodeErrOp : I32EnumAttrCase<"MPI_ERR_OP", 33, "MPI_ERR_OP">;
def MpiCodeErrOther : I32EnumAttrCase<"MPI_ERR_OTHER", 34, "MPI_ERR_OTHER">;
def MpiCodeErrPending : I32EnumAttrCase<"MPI_ERR_PENDING", 35, "MPI_ERR_PENDING">;
def MpiCodeErrPort : I32EnumAttrCase<"MPI_ERR_PORT", 36, "MPI_ERR_PORT">;
def MpiCodeErrProcAborted : I32EnumAttrCase<"MPI_ERR_PROC_ABORTED", 37, "MPI_ERR_PROC_ABORTED">;
def MpiCodeErrQuota : I32EnumAttrCase<"MPI_ERR_QUOTA", 38, "MPI_ERR_QUOTA">;
def MpiCodeErrRank : I32EnumAttrCase<"MPI_ERR_RANK", 39, "MPI_ERR_RANK">;
def MpiCodeErrReadOnly : I32EnumAttrCase<"MPI_ERR_READ_ONLY", 40, "MPI_ERR_READ_ONLY">;
def MpiCodeErrRequest : I32EnumAttrCase<"MPI_ERR_REQUEST", 41, "MPI_ERR_REQUEST">;
def MpiCodeErrRmaAttach : I32EnumAttrCase<"MPI_ERR_RMA_ATTACH", 42, "MPI_ERR_RMA_ATTACH">;
def MpiCodeErrRmaConflict : I32EnumAttrCase<"MPI_ERR_RMA_CONFLICT", 43, "MPI_ERR_RMA_CONFLICT">;
def MpiCodeErrRmaFlavor : I32EnumAttrCase<"MPI_ERR_RMA_FLAVOR", 44, "MPI_ERR_RMA_FLAVOR">;
def MpiCodeErrRmaRange : I32EnumAttrCase<"MPI_ERR_RMA_RANGE", 45, "MPI_ERR_RMA_RANGE">;
def MpiCodeErrRmaShared : I32EnumAttrCase<"MPI_ERR_RMA_SHARED", 46, "MPI_ERR_RMA_SHARED">;
def MpiCodeErrRmaSync : I32EnumAttrCase<"MPI_ERR_RMA_SYNC", 47, "MPI_ERR_RMA_SYNC">;
def MpiCodeErrRoot : I32EnumAttrCase<"MPI_ERR_ROOT", 48, "MPI_ERR_ROOT">;
def MpiCodeErrService : I32EnumAttrCase<"MPI_ERR_SERVICE", 49, "MPI_ERR_SERVICE">;
def MpiCodeErrSession : I32EnumAttrCase<"MPI_ERR_SESSION", 50, "MPI_ERR_SESSION">;
def MpiCodeErrSize : I32EnumAttrCase<"MPI_ERR_SIZE", 51, "MPI_ERR_SIZE">;
def MpiCodeErrSpawn : I32EnumAttrCase<"MPI_ERR_SPAWN", 52, "MPI_ERR_SPAWN">;
def MpiCodeErrTag : I32EnumAttrCase<"MPI_ERR_TAG", 53, "MPI_ERR_TAG">;
def MpiCodeErrTopology : I32EnumAttrCase<"MPI_ERR_TOPOLOGY", 54, "MPI_ERR_TOPOLOGY">;
def MpiCodeErrTruncate : I32EnumAttrCase<"MPI_ERR_TRUNCATE", 55, "MPI_ERR_TRUNCATE">;
def MpiCodeErrType : I32EnumAttrCase<"MPI_ERR_TYPE", 56, "MPI_ERR_TYPE">;
def MpiCodeErrUnknown : I32EnumAttrCase<"MPI_ERR_UNKNOWN", 57, "MPI_ERR_UNKNOWN">;
def MpiCodeErrUnsupportedDatarep : I32EnumAttrCase<"MPI_ERR_UNSUPPORTED_DATAREP", 58, "MPI_ERR_UNSUPPORTED_DATAREP">;
def MpiCodeErrUnsupportedOperation : I32EnumAttrCase<"MPI_ERR_UNSUPPORTED_OPERATION", 59, "MPI_ERR_UNSUPPORTED_OPERATION">;
def MpiCodeErrValueTooLarge : I32EnumAttrCase<"MPI_ERR_VALUE_TOO_LARGE", 60, "MPI_ERR_VALUE_TOO_LARGE">;
def MpiCodeErrWin : I32EnumAttrCase<"MPI_ERR_WIN", 61, "MPI_ERR_WIN">;
def MpiCodeErrLastcode : I32EnumAttrCase<"MPI_ERR_LASTCODE", 62, "MPI_ERR_LASTCODE">;

def MpiErrorClassEnum : I32EnumAttr<"MpiErrorClassEnum", 
                                "MPI error class name",
  [ MpiCodeSuccess
    ,MpiCodeErrAccess
    ,MpiCodeErrAmode
    ,MpiCodeErrArg
    ,MpiCodeErrAssert
    ,MpiCodeErrBadFile
    ,MpiCodeErrBase
    ,MpiCodeErrBuffer
    ,MpiCodeErrComm
    ,MpiCodeErrConversion
    ,MpiCodeErrCount
    ,MpiCodeErrDims
    ,MpiCodeErrDisp
    ,MpiCodeErrDupDatarep
    ,MpiCodeErrErrhandler
    ,MpiCodeErrFile
    ,MpiCodeErrFileExists
    ,MpiCodeErrFileInUse
    ,MpiCodeErrGroup
    ,MpiCodeErrInfo
    ,MpiCodeErrInfoKey
    ,MpiCodeErrInfoNokey
    ,MpiCodeErrInfoValue
    ,MpiCodeErrInStatus
    ,MpiCodeErrIntern
    ,MpiCodeErrIo
    ,MpiCodeErrKeyval
    ,MpiCodeErrLocktype
    ,MpiCodeErrName
    ,MpiCodeErrNoMem
    ,MpiCodeErrNoSpace
    ,MpiCodeErrNoSuchFile
    ,MpiCodeErrNotSame
    ,MpiCodeErrOp
    ,MpiCodeErrOther
    ,MpiCodeErrPending
    ,MpiCodeErrPort
    ,MpiCodeErrProcAborted
    ,MpiCodeErrQuota
    ,MpiCodeErrRank
    ,MpiCodeErrReadOnly
    ,MpiCodeErrRequest
    ,MpiCodeErrRmaAttach
    ,MpiCodeErrRmaConflict
    ,MpiCodeErrRmaFlavor
    ,MpiCodeErrRmaRange
    ,MpiCodeErrRmaShared
    ,MpiCodeErrRmaSync
    ,MpiCodeErrRoot
    ,MpiCodeErrService
    ,MpiCodeErrSession
    ,MpiCodeErrSize
    ,MpiCodeErrSpawn
    ,MpiCodeErrTag
    ,MpiCodeErrTopology
    ,MpiCodeErrTruncate
    ,MpiCodeErrType
    ,MpiCodeErrUnknown
    ,MpiCodeErrUnsupportedDatarep
    ,MpiCodeErrUnsupportedOperation
    ,MpiCodeErrValueTooLarge
    ,MpiCodeErrWin
    ,MpiCodeErrLastcode]> {
  let genSpecializedAttr = 0;
  let cppNamespace = "::mlir::mpi";
}

def MpiErrorClassAttr : EnumAttr<MPI_Dialect, MpiErrorClassEnum, "errclass"> {
  let assemblyFormat = "`<` $value `>`";
}

#endif // MLIR_DIALECT_MPI_IR_MPI
