//******************************************************************************
//
// File:    BigInt.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.BigInt
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

import edu.rit.util.Random;
import java.util.Arrays;

/**
 * Class BigInt provides an arbitrary-size unsigned integer. Internally, a
 * bigint is represented as an array of 32-bit words containing the bigint's
 * value in little-endian order.
 *
 * @author  Alan Kaminsky
 * @version 24-Oct-2017
 */
public class BigInt
	{

// Hidden constants.

	private static final long MASK32 = 0xFFFFFFFFL;

// Exported data members.

	/**
	 * Array of 32-bit words containing this bigint's value in little-endian
	 * order.
	 */
	public final int[] value;

// Exported constructors.

	/**
	 * Construct a new bigint with the value 0.
	 *
	 * @param  bitSize  Bit size.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bitSize</TT> &lt; 0.
	 */
	public BigInt
		(int bitSize)
		{
		if (bitSize < 0)
			throw new IllegalArgumentException (String.format
				("BigInt(): bitSize = %d illegal", bitSize));
		value = new int [wordSize (bitSize)];
		}

	/**
	 * Construct a new bigint that is a copy of the given bigint.
	 *
	 * @param  bigint  BigInt.
	 */
	public BigInt
		(BigInt bigint)
		{
		value = (int[]) bigint.value.clone();
		}

// Exported operations.

	/**
	 * Returns this bigint's word size. This is the number of 32-bit words
	 * required to hold this bigint's value.
	 *
	 * @return  Word size.
	 */
	public int wordSize()
		{
		return value.length;
		}

	/**
	 * Returns this bigint's bit size. This is the word size times 32.
	 *
	 * @return  Bit size.
	 */
	public int bitSize()
		{
		return wordSize()*32;
		}

	/**
	 * Set this bigint to a randomly chosen value. The value is chosen uniformly
	 * at random in the range 0 to 2<SUP><I>B</I></SUP>&minus;1, where <I>B</I>
	 * is this bigint's bit size.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  This bigint.
	 */
	public BigInt randomize
		(Random prng)
		{
		for (int i = 0; i < value.length; ++ i)
			value[i] = prng.nextInteger();
		return this;
		}

	/**
	 * Set this bigint to a randomly chosen value in the given range. The value
	 * is chosen uniformly at random in the range 0 to
	 * 2<SUP><I>B</I></SUP>&minus;1, where <I>B</I>
	 * is the smaller of the argument <TT>B</TT> and this bigint's bit size.
	 *
	 * @param  prng  Pseudorandom number generator.
	 * @param  B     Number of bits to choose at random.
	 *
	 * @return  This bigint.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>B</TT> &lt; 0.
	 */
	public BigInt randomize
		(Random prng,
		 int B)
		{
		if (B < 0)
			throw new IllegalArgumentException (String.format
				("BigInt.randomize(): B = %d illegal", B));
		B = Math.min (B, bitSize());
		int i = 0;
		while (B >= 32)
			{
			value[i] = prng.nextInteger();
			B -= 32;
			++ i;
			}
		if (B > 0)
			{
			value[i] = prng.nextInteger() & ((1 << B) - 1);
			++ i;
			}
		while (i < value.length)
			{
			value[i] = 0;
			++ i;
			}
		return this;
		}

	/**
	 * Set this bigint's value from the given string.
	 *
	 * @param  s  Hexadecimal value. The hexadecimal bytes of this string are in
	 *            big-endian order.
	 *
	 * @return  This bigint.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>s</TT> is null.
	 * @exception  NumberFormatException
	 *     (unchecked exception) Thrown if <TT>s</TT> is not a hexadecimal
	 *     value.
	 */
	public BigInt fromString
		(String s)
		{
		zero();
		int n = s.length();
		for (int i = 0; i < n; ++ i)
			{
			shiftLeft4();
			value[0] |= charToHex (s.charAt (i));
			}
		return this;
		}

	/**
	 * Convert this bigint's value to a string. The hexadecimal bytes of this
	 * bigint are displayed in big-endian order.
	 *
	 * @return  String value.
	 */
	public String toString()
		{
		StringBuilder b = new StringBuilder();
		for (int i = 0; i < value.length; ++ i)
			for (int j = 0; j < 32; j += 4)
				b.insert (0, hexToChar ((value[i] >> j) & 15));
		return b.toString();
		}

	/**
	 * Set this bigint's value to the given unsigned integer.
	 *
	 * @param  v  Unsigned integer value.
	 *
	 * @return  This bigint, with its value set to <TT>v</TT>.
	 */
	public BigInt assign
		(int v)
		{
		value[0] = v;
		for (int i = 1; i < value.length; ++ i)
			value[i] = 0;
		return this;
		}

	/**
	 * Set this bigint's value to that of the given bigint.
	 *
	 * @param  bigint  Bigint.
	 *
	 * @return  This bigint, with its value set to that of <TT>bigint</TT>.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bigint</TT>'s bit size is not the
	 *     same as this bigint's bit size.
	 */
	public BigInt assign
		(BigInt bigint)
		{
		if (bigint.value.length != this.value.length)
			throw new IllegalArgumentException
				("BigInt.assign(): Different word sizes");
		System.arraycopy (bigint.value, 0, this.value, 0, this.value.length);
		return this;
		}

	/**
	 * Increment this bigint.
	 *
	 * @return  This bigint, with its value increased by 1.
	 */
	public BigInt increment()
		{
		long acc = 1;
		for (int i = 0; i < value.length; ++ i)
			{
			acc += value[i] & MASK32;
			value[i] = (int) acc;
			acc >>= 32;
			}
		return this;
		}

	/**
	 * Add the given bigint to this bigint.
	 *
	 * @param  bigint  Bigint.
	 *
	 * @return  This bigint, with its value increased by <TT>bigint</TT>.
	 */
	public BigInt add
		(BigInt bigint)
		{
		long acc = 0;
		for (int i = 0; i < value.length; ++ i)
			{
			acc = acc + (this.value[i] & MASK32) + (bigint.value[i] & MASK32);
			this.value[i] = (int) acc;
			acc >>= 32;
			}
		return this;
		}

	/**
	 * Exclusive-or this bigint with the given integer.
	 *
	 * @param  x  Integer.
	 *
	 * @return  This bigint, exclusive-ored with <TT>x</TT>.
	 */
	public BigInt xor
		(int x)
		{
		value[0] ^= x;
		return this;
		}

	/**
	 * Exclusive-or the given bigint into this bigint.
	 *
	 * @param  bigint  Bigint.
	 *
	 * @return  This bigint, exclusive-ored with <TT>bigint</TT>.
	 */
	public BigInt xor
		(BigInt bigint)
		{
		for (int i = 0; i < value.length; ++ i)
			this.value[i] ^= bigint.value[i];
		return this;
		}

	/**
	 * Complement this bigint.
	 *
	 * @return  This bigint, with each bit flipped.
	 */
	public BigInt complement()
		{
		for (int i = 0; i < value.length; ++ i)
			value[i] = ~value[i];
		return this;
		}

	/**
	 * Get the bit at the given position in this bigint.
	 *
	 * @param  i  Bit position.
	 *
	 * @return  Bit at position <TT>i</TT> (0 or 1).
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
	 */
	public int getBit
		(int i)
		{
		return (value[i >> 5] >> (i & 31)) & 1;
		}

	/**
	 * Put the given value into the bit at the given position in this bigint.
	 *
	 * @param  i  Bit position.
	 * @param  v  Bit value (0 or 1).
	 *
	 * @return  This bigint, with bit position <TT>i</TT> set to <TT>v</TT>.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
	 */
	public BigInt putBit
		(int i,
		 int v)
		{
		if (v == 0)
			value[i >> 5] &= ~(1 << (i & 31));
		else
			value[i >> 5] |=  (1 << (i & 31));
		return this;
		}

	/**
	 * Flip the bit at the given position in this bigint.
	 *
	 * @param  i  Bit position.
	 *
	 * @return  This bigint, with bit position <TT>i</TT> flipped.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>i</TT> is out of bounds.
	 */
	public BigInt flipBit
		(int i)
		{
		value[i >> 5] ^= (1 << (i & 31));
		return this;
		}

	/**
	 * Scatter bits of the given long integer into the given positions in this
	 * bigint. Bit position 0 of <TT>val</TT> goes into bit position
	 * <TT>pos[0]</TT> of this bigint, bit position 1 of <TT>val</TT> goes into
	 * bit position <TT>pos[1]</TT> of this bigint, and so on.
	 *
	 * @param  pos  Array of bit positions in this bigint.
	 * @param  val  Long integer to scatter.
	 *
	 * @return  This bigint, as modified.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if any bit position in <TT>pos</TT> is
	 *     out of bounds.
	 */
	public BigInt scatterBits
		(int[] pos,
		 long val)
		{
		for (int i = 0; i < pos.length; ++ i)
			{
			putBit (pos[i], (int)(val & 1));
			val >>>= 1;
			}
		return this;
		}

	/**
	 * Scatter bits of the given bigint into the given positions in this bigint.
	 * Bit position 0 of <TT>val</TT> goes into bit position <TT>pos[0]</TT> of
	 * this bigint, bit position 1 of <TT>val</TT> goes into bit position
	 * <TT>pos[1]</TT> of this bigint, and so on.
	 *
	 * @param  pos  Array of bit positions in this bigint.
	 * @param  val  Bigint to scatter.
	 *
	 * @return  This bigint, as modified.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if any bit position in <TT>pos</TT> is
	 *     out of bounds.
	 */
	public BigInt scatterBits
		(int[] pos,
		 BigInt val)
		{
		for (int i = 0; i < pos.length; ++ i)
			putBit (pos[i], val.getBit (i));
		return this;
		}

	/**
	 * Left-shift this bigint.
	 *
	 * @param  shift  Number of bit positions to shift.
	 *
	 * @return  This bigint, left-shifted by <TT>shift</TT> bits.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInt leftShift
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInt.leftShift(): shift = %d illegal", shift));
		else if (shift >= value.length*32)
			zero();
		else if (shift > 0)
			leftShift (value, shift);
		return this;
		}

	/**
	 * Right-shift this bigint.
	 *
	 * @param  shift  Number of bit positions to shift.
	 *
	 * @return  This bigint, right-shifted by <TT>shift</TT> bits.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>shift</TT> &lt; 0.
	 */
	public BigInt rightShift
		(int shift)
		{
		if (shift < 0)
			throw new IllegalArgumentException (String.format
				("BigInt.rightShift(): shift = %d illegal", shift));
		else if (shift >= value.length*32)
			zero();
		else if (shift > 0)
			rightShift (value, shift);
		return this;
		}

	/**
	 * Pack the given integer array into this bigint in little-endian order.
	 * Elements of the <TT>array</TT> at indexes 0 through <I>L</I>&minus;1 are
	 * copied into words of this bigint at indexes 0 through <I>L</I>&minus;1,
	 * where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void pack
		(int[] array)
		{
		pack (array, 0);
		}

	/**
	 * Pack a portion of the given integer array into this bigint in
	 * little-endian order. Elements of the <TT>array</TT> at indexes
	 * <TT>off</TT> through <TT>off</TT>+<I>L</I>&minus;1 are copied into words
	 * of this bigint at indexes 0 through <I>L</I>&minus;1, where <I>L</I> is
	 * this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to pack.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void pack
		(int[] array,
		 int off)
		{
		if (off < 0 || off + value.length > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.pack(): off = %d out of bounds", off));
		System.arraycopy (array, off, value, 0, value.length);
		}

	/**
	 * Unpack this bigint into the given integer array in little-endian order.
	 * Words of this bigint at indexes 0 through <I>L</I>&minus;1 are copied
	 * into elements of the <TT>array</TT> at indexes 0 through
	 * <I>L</I>&minus;1, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void unpack
		(int[] array)
		{
		unpack (array, 0);
		}

	/**
	 * Unpack this bigint into a portion of the given integer array in
	 * little-endian order. Words of this bigint at indexes 0 through
	 * <I>L</I>&minus;1 are copied into elements of the <TT>array</TT> at
	 * indexes <TT>off</TT> through <TT>off</TT>+<I>L</I>&minus;1, where
	 * <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to unpack.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void unpack
		(int[] array,
		 int off)
		{
		if (off < 0 || off + value.length > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.unpack(): off = %d out of bounds", off));
		System.arraycopy (value, 0, array, off, value.length);
		}

	/**
	 * Pack the given integer array into this bigint in big-endian order.
	 * Elements of the <TT>array</TT> at indexes 0 through <I>L</I>&minus;1 are
	 * copied into words of this bigint at indexes <I>L</I>&minus;1 through 0,
	 * where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void packBigEndian
		(int[] array)
		{
		packBigEndian (array, 0);
		}

	/**
	 * Pack a portion of the given integer array into this bigint in big-endian
	 * order. Elements of the <TT>array</TT> at indexes <TT>off</TT> through
	 * <TT>off</TT>+<I>L</I>&minus;1 are copied into words of this bigint at
	 * indexes <I>L</I>&minus;1 through 0, where <I>L</I> is this bigint's word
	 * size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to pack.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void packBigEndian
		(int[] array,
		 int off)
		{
		if (off < 0 || off + value.length > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.packBigEndian(): off = %d out of bounds", off));
		for (int i = value.length - 1, j = off; i >= 0; -- i, ++ j)
			value[i] = array[j];
		}

	/**
	 * Unpack this bigint into the given integer array in big-endian order.
	 * Words of this bigint at indexes <I>L</I>&minus;1 through 0 are copied
	 * into elements of the <TT>array</TT> at indexes 0 through
	 * <I>L</I>&minus;1, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void unpackBigEndian
		(int[] array)
		{
		unpackBigEndian (array, 0);
		}

	/**
	 * Unpack this bigint into a portion of the given integer array in
	 * big-endian order. Words of this bigint at indexes <I>L</I>&minus;1
	 * through 0 are copied into elements of the <TT>array</TT> at indexes
	 * <TT>off</TT> through <TT>off</TT>+<I>L</I>&minus;1, where <I>L</I> is
	 * this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to unpack.
	 *
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I> &gt; <TT>array.length</TT>.
	 */
	public void unpackBigEndian
		(int[] array,
		 int off)
		{
		if (off < 0 || off + value.length > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.unpackBigEndian(): off = %d out of bounds", off));
		for (int i = value.length - 1, j = off; i >= 0; -- i, ++ j)
			array[j] = value[i];
		}

	/**
	 * Pack the given long integer array into this bigint in little-endian
	 * order. Elements of the <TT>array</TT> at indexes 0 through
	 * <I>L</I>/2&minus;1 are copied into words of this bigint at indexes 0
	 * through <I>L</I>&minus;1, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I>/2 &gt;
	 *     <TT>array.length</TT>.
	 */
	public void pack
		(long[] array)
		{
		pack (array, 0);
		}

	/**
	 * Pack a portion of the given long integer array into this bigint in
	 * little-endian order. Elements of the <TT>array</TT> at indexes
	 * <TT>off</TT> through <TT>off</TT>+<I>L</I>/2&minus;1 are copied into
	 * words of this bigint at indexes 0 through <I>L</I>&minus;1, where
	 * <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to pack.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I>/2 &gt; <TT>array.length</TT>.
	 */
	public void pack
		(long[] array,
		 int off)
		{
		if ((value.length & 1) != 0)
			throw new IllegalArgumentException (String.format
				("BigInt.pack(): word size = %d illegal", value.length));
		int N = value.length/2;
		if (off < 0 || off + N > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.pack(): off = %d out of bounds", off));
		for (int i = 0; i < N; ++ i)
			{
			value[2*i  ] = (int)(array[off+i]);
			value[2*i+1] = (int)(array[off+i] >> 32);
			}
		}

	/**
	 * Unpack this bigint into the given long integer array in little-endian
	 * order. Words of this bigint at indexes 0 through <I>L</I>&minus;1 are
	 * copied into elements of the <TT>array</TT> at indexes 0 through
	 * <I>L</I>/2&minus;1, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I>/2 &gt;
	 *     <TT>array.length</TT>.
	 */
	public void unpack
		(long[] array)
		{
		unpack (array, 0);
		}

	/**
	 * Unpack this bigint into a portion of the given long integer array in
	 * little-endian order. Words of this bigint at indexes 0 through
	 * <I>L</I>&minus;1 are copied into elements of the <TT>array</TT> at
	 * indexes <TT>off</TT> through <TT>off</TT>+<I>L</I>/2&minus;1, where
	 * <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to unpack.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I>/2 &gt; <TT>array.length</TT>.
	 */
	public void unpack
		(long[] array,
		 int off)
		{
		if ((value.length & 1) != 0)
			throw new IllegalArgumentException (String.format
				("BigInt.unpack(): word size = %d illegal", value.length));
		int N = value.length/2;
		if (off < 0 || off + N > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.unpack(): off = %d out of bounds", off));
		for (int i = 0; i < N; ++ i)
			array[off+i] = ((long)(value[2*i]) & MASK32) | 
				((long)(value[2*i+1]) << 32);
		}

	/**
	 * Pack the given long integer array into this bigint in big-endian order.
	 * Elements of the <TT>array</TT> at indexes 0 through <I>L</I>/2&minus;1
	 * are copied into words of this bigint at indexes <I>L</I>&minus;1 through
	 * 0, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I>/2 &gt;
	 *     <TT>array.length</TT>.
	 */
	public void packBigEndian
		(long[] array)
		{
		packBigEndian (array, 0);
		}

	/**
	 * Pack a portion of the given long integer array into this bigint in
	 * big-endian order. Elements of the <TT>array</TT> at indexes <TT>off</TT>
	 * through <TT>off</TT>+<I>L</I>/2&minus;1 are copied into words of this
	 * bigint at indexes <I>L</I>&minus;1 through 0, where <I>L</I> is this
	 * bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to pack.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I>/2 &gt; <TT>array.length</TT>.
	 */
	public void packBigEndian
		(long[] array,
		 int off)
		{
		if ((value.length & 1) != 0)
			throw new IllegalArgumentException (String.format
				("BigInt.packBigEndian(): word size = %d illegal",
				 value.length));
		int N = value.length/2;
		if (off < 0 || off + N > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.packBigEndian(): off = %d out of bounds", off));
		for (int i = 0; i < N; ++ i)
			{
			value[value.length-2*i-1] = (int)(array[off+i] >> 32);
			value[value.length-2*i-2] = (int)(array[off+i]);
			}
		}

	/**
	 * Unpack this bigint into the given long integer array in big-endian order.
	 * Words of this bigint at indexes <I>L</I>&minus;1 through 0 are copied
	 * into elements of the <TT>array</TT> at indexes 0 through
	 * <I>L</I>/2&minus;1, where <I>L</I> is this bigint's word size.
	 *
	 * @param  array  Integer array.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <I>L</I>/2 &gt;
	 *     <TT>array.length</TT>.
	 */
	public void unpackBigEndian
		(long[] array)
		{
		unpackBigEndian (array, 0);
		}

	/**
	 * Unpack this bigint into a portion of the given long integer array in
	 * big-endian order. Words of this bigint at indexes <I>L</I>&minus;1
	 * through 0 are copied into elements of the <TT>array</TT> at indexes
	 * <TT>off</TT> through <TT>off</TT>+<I>L</I>/2&minus;1, where <I>L</I> is
	 * this bigint's word size.
	 *
	 * @param  array  Integer array.
	 * @param  off    First array index to unpack.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <I>L</I> is not a multiple of 2.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0 or
	 *     <TT>off</TT>+<I>L</I>/2 &gt; <TT>array.length</TT>.
	 */
	public void unpackBigEndian
		(long[] array,
		 int off)
		{
		if ((value.length & 1) != 0)
			throw new IllegalArgumentException (String.format
				("BigInt.unpackBigEndian(): word size = %d illegal",
				 value.length));
		int N = value.length/2;
		if (off < 0 || off + N > array.length)
			throw new IndexOutOfBoundsException (String.format
				("BigInt.unpackBigEndian(): off = %d out of bounds", off));
		for (int i = 0; i < N; ++ i)
			array[off+i] = ((long)(value[value.length-2*i-2]) & MASK32) | 
				((long)(value[value.length-2*i-1]) << 32);
		}

	/**
	 * Determine if this bigint's value equals the given bigint's value.
	 *
	 * @param  bigint  Bigint.
	 *
	 * @return  True if values are equal, false if not.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>bigint</TT>'s bit size is not the
	 *     same as this bigint's bit size.
	 */
	public boolean eq
		(BigInt bigint)
		{
		if (bigint.value.length != this.value.length)
			throw new IllegalArgumentException
				("BigInt.assign(): Different word sizes");
		for (int i = 0; i < this.value.length; ++ i)
			if (this.value[i] != bigint.value[i])
				return false;
		return true;
		}

// Hidden operations.

	/**
	 * Convert bit size to word size.
	 *
	 * @param  bitSize  Bit size.
	 *
	 * @return  Number of words needed to hold <TT>bitSize</TT> bits.
	 */
	static int wordSize
		(int bitSize)
		{
		return (bitSize + 31) >> 5;
		}

	/**
	 * Set this bigint to a value of 0.
	 */
	private void zero()
		{
		Arrays.fill (value, 0);
		}

	/**
	 * Shift this bigint left 4 bit positions.
	 */
	private void shiftLeft4()
		{
		for (int i = value.length - 1; i > 0; -- i)
			value[i] = (value[i] << 4) | (value[i-1] >>> 28);
		value[0] = value[0] << 4;
		}

	/**
	 * Convert the given character to a hexadecimal digit.
	 */
	private int charToHex
		(char c)
		{
		if ('0' <= c && c <= '9')
			return c - '0';
		else if ('A' <= c && c <= 'F')
			return c - 'A' + 10;
		else if ('a' <= c && c <= 'f')
			return c - 'a' + 10;
		else
			throw new NumberFormatException (String.format
				("Illegal hex digit '%c'", c));
		}

	/**
	 * Convert the given hexadecimal digit to a character.
	 */
	private char hexToChar
		(int d)
		{
		if (d <= 9)
			return (char)(d + '0');
		else
			return (char)(d - 10 + 'a');
		}

	/**
	 * Left-shift the given array by the given number of bits.
	 */
	private static void leftShift
		(int[] a,
		 int n)
		{
		int off = n >>> 5; // off = number of words to left-shift
		n &= 31;           // n = number of bits within word to left-shift
		int m = 32 - n;    // m = number of bits within word to right-shift

		int d = a.length - 1;
		int s = d - off;
		if (n == 0)
			{
			// Shift entire words only.
			while (s >= 0)
				{
				a[d] = a[s];
				-- d;
				-- s;
				}
			}
		else
			{
			// Shift words and bits within words.
			while (s > 0)
				{
				a[d] = (a[s] << n) | (a[s-1] >>> m);
				-- d;
				-- s;
				}
			a[d] = (a[s] << n);
			-- d;
			}

		// Clear remaining least significant words.
		while (d >= 0)
			{
			a[d] = 0;
			-- d;
			}
		}

	/**
	 * Right-shift the given array by the given number of bits.
	 */
	private static void rightShift
		(int[] a,
		 int n)
		{
		int off = n >>> 5; // off = number of words to right-shift
		n &= 31;           // n = number of bits within word to right-shift
		int m = 32 - n;    // m = number of bits within word to left-shift
		int alenm1 = a.length - 1;

		int d = 0;
		int s = off;
		if (n == 0)
			{
			// Shift entire words only.
			while (s <= alenm1)
				{
				a[d] = a[s];
				++ d;
				++ s;
				}
			}
		else
			{
			// Shift words and bits within words.
			while (s < alenm1)
				{
				a[d] = (a[s] >>> n) | (a[s+1] << m);
				++ d;
				++ s;
				}
			a[d] = (a[s] >>> n);
			++ d;
			}

		// Clear remaining most significant words.
		while (d <= alenm1)
			{
			a[d] = 0;
			++ d;
			}
		}

	}
