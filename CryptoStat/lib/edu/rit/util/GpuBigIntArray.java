//******************************************************************************
//
// File:    GpuBigIntArray.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.GpuBigIntArray
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the CryptoStat Library ("CryptoStat").
// CryptoStat is free software; you can redistribute it and/or modify it under
// the terms of the GNU General Public License as published by the Free Software
// Foundation; either version 3 of the License, or (at your option) any later
// version.
//
// CryptoStat is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.util;

import edu.rit.gpu.Gpu;
import edu.rit.gpu.GpuIntArray;
import edu.rit.gpu.Module;

/**
 * Class GpuBigIntArray provides an array of {@linkplain BigInt}s located on the
 * GPU and mirrored on the CPU.
 *
 * @author  Alan Kaminsky
 * @version 23-Aug-2017
 */
public class GpuBigIntArray
	{

// Exported data members.

	/**
	 * Array of {@linkplain BigInt}s.
	 */
	public final BigInt[] item;

// Hidden data members.

	private int wordSize;
	private int len;
	private GpuIntArray array;

// Exported constructors.

	/**
	 * Construct a new bigint array with the given length. Each element of the
	 * array is initialized to a bigint with the given bit size and a value of
	 * 0.
	 * <P>
	 * <I>Note:</I> After constructing, call the <TT>allocate()</TT> or
	 * <TT>mirror()</TT> method to set up storage on the GPU.
	 *
	 * @param  bitSize  Bit size.
	 * @param  len      Array length.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bitSize</TT> &lt; 0.
	 * @exception  NegativeArraySizeException
	 *     (unchecked exception) Thrown if <TT>len</TT> &lt; 0.
	 */
	public GpuBigIntArray
		(int bitSize,
		 int len)
		{
		this.item = new BigInt [len];
		for (int i = 0; i < len; ++ i)
			this.item[i] = new BigInt (bitSize);
		this.wordSize = BigInt.wordSize (bitSize);
		this.len = len;
		}

// Exported operations.

	/**
	 * Clear this bigint array. All elements are set to 0.
	 */
	public void clear()
		{
		for (int i = 0; i < len; ++ i)
			item[i].assign (0);
		}

	/**
	 * Allocate storage for this bigint array in global memory on the GPU. This
	 * bigint array will mirror the allocated storage.
	 *
	 * @param  gpu  GPU accelerator.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void allocate
		(Gpu gpu)
		{
		array = gpu.getIntArray (len*wordSize);
		}

	/**
	 * Set this bigint array to mirror the given global variable on the GPU.
	 * <P>
	 * <I>Note:</I> It is assumed that the global variable is a uint32_t array
	 * with enough elements to hold this bigint array.
	 *
	 * @param  module  Kernel module object.
	 * @param  name    Global variable name.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void mirror
		(Module module,
		 String name)
		{
		array = module.getIntArray (name, len*wordSize);
		}

	/**
	 * Get the underlying <TT>int</TT> array on the GPU.
	 *
	 * @return  Underlying <TT>int</TT> array.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if the <TT>allocate()</TT> or
	 *     <TT>mirror()</TT> method has not been called.
	 */
	public GpuIntArray array()
		{
		verifyArray();
		return array;
		}

	/**
	 * Copy this GPU bigint array from the host CPU's memory to the given GPU
	 * device's memory.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if the <TT>allocate()</TT> or
	 *     <TT>mirror()</TT> method has not been called.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void hostToDev()
		{
		verifyArray();
		int off = 0;
		for (int i = 0; i < len; ++ i)
			{
			item[i].unpack (array.item, off);
			off += wordSize;
			}
		array.hostToDev();
		}

	/**
	 * Copy this GPU bigint array from the given GPU device's memory to the host
	 * CPU's memory.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if the <TT>allocate()</TT> or
	 *     <TT>mirror()</TT> method has not been called.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	public void devToHost()
		{
		verifyArray();
		array.devToHost();
		int off = 0;
		for (int i = 0; i < len; ++ i)
			{
			item[i].pack (array.item, off);
			off += wordSize;
			}
		}

	/**
	 * Returns the word size of the bigints in this GPU bigint array. This is
	 * the number of 32-bit words required to hold each bigint's value.
	 *
	 * @return  Word size.
	 */
	public int wordSize()
		{
		return wordSize;
		}

	/**
	 * Returns the bit size of the bigints in this GPU bigint array. This is the
	 * word size times 32.
	 *
	 * @return  Bit size.
	 */
	public int bitSize()
		{
		return wordSize()*32;
		}

// Hidden operations.

	/**
	 * Verify that the underlying <TT>int</TT> array has been allocated on the
	 * GPU.
	 */
	private void verifyArray()
		{
		if (array == null)
			throw new NullPointerException
				("GpuBigIntArray.verifyArray(): Not allocated on GPU");
		}

	}
