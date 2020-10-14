/*************************************************************************
 * Copyright (c) 2015-2019, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclAllReduce, const void* sendbuff, void* recvbuff, void* tempbuff1, void* tempbuff2, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, void* tempbuff1, void* tempbuff2, size_t count,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  struct ncclInfo info = { ncclCollAllReduce, "AllReduce",
    sendbuff, recvbuff, tempbuff1, tempbuff2, count, datatype, op, 0, comm, stream, /* Args */
    ALLREDUCE_CHUNKSTEPS, ALLREDUCE_SLICESTEPS};
  return ncclEnqueueCheck(&info);
}
