//******************************************************************************
//
// File:    CtxSched.java
// Package: edu.rit.gpu
// Unit:    Enum edu.rit.gpu.CtxSched
//
// This Java source file is copyright (C) 2014 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.gpu;

/**
 * Enum CtxSched enumerates GPU context scheduling flags.
 *
 * @author  Alan Kaminsky
 * @version 31-Jan-2017
 */
public enum CtxSched
	{

	/**
	 * Instruct CUDA to actively spin when waiting for results from the GPU.
	 * This can decrease latency when waiting for the GPU, but might lower the
	 * performance of CPU threads if they are performing work in parallel with
	 * the CUDA thread.
	 */
	CU_CTX_SCHED_SPIN (Cuda.J_CU_CTX_SCHED_SPIN),

	/**
	 * Instruct CUDA to yield its thread when waiting for results from the GPU.
	 * This can increase latency when waiting for the GPU, but can increase the
	 * performance of CPU threads performing work in parallel with the GPU.
	 */
	CU_CTX_SCHED_YIELD (Cuda.J_CU_CTX_SCHED_YIELD),

	/**
	 * Instruct CUDA to block the CPU thread on a synchronization primitive when
	 * waiting for the GPU to finish work.
	 */
	CU_CTX_SCHED_BLOCKING_SYNC (Cuda.J_CU_CTX_SCHED_BLOCKING_SYNC),

	/**
	 * Instruct CUDA to determine the context scheduling automatically. Uses a
	 * heuristic based on the number of active CUDA contexts in the process
	 * <I>C</I> and the number of logical processors in the system <I>P</I>. If
	 * <I>C</I> &gt; <I>P</I>, then CUDA will yield to other OS threads when
	 * waiting for the GPU (CU_CTX_SCHED_YIELD), otherwise CUDA will not yield
	 * while waiting for results and actively spin on the processor
	 * (CU_CTX_SCHED_SPIN). However, on low power devices like Tegra, it always
	 * defaults to CU_CTX_SCHED_BLOCKING_SYNC.
	 */
	CU_CTX_SCHED_AUTO (Cuda.J_CU_CTX_SCHED_AUTO);

	/**
	 * Enumeral value.
	 */
	public final int value;

	/**
	 * Construct a new CtxSched enumeral.
	 *
	 * @param  value  Enumeral value.
	 */
	private CtxSched
		(int value)
		{
		this.value = value;
		}

	/**
	 * Convert the given enumeral value to an enumeral.
	 *
	 * @param  value  Enumeral value.
	 *
	 * @return  Enumeral.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>value</TT> is illegal.
	 */
	public static CtxSched of
		(int value)
		{
		switch (value)
			{
			case Cuda.J_CU_CTX_SCHED_SPIN:
				return CU_CTX_SCHED_SPIN;
			case Cuda.J_CU_CTX_SCHED_YIELD:
				return CU_CTX_SCHED_YIELD;
			case Cuda.J_CU_CTX_SCHED_BLOCKING_SYNC:
				return CU_CTX_SCHED_BLOCKING_SYNC;
			case Cuda.J_CU_CTX_SCHED_AUTO:
				return CU_CTX_SCHED_AUTO;
			default:
				throw new IllegalArgumentException (String.format
					("CtxSched.of(): value = %d illegal", value));
			}
		}

	}
