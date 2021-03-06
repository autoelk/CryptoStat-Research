//******************************************************************************
//
// File:    BitSet.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.BitSet
//
// This Java source file is copyright (C) 2018 by Alan Kaminsky. All rights
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

package edu.rit.util;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.io.Streamable;
import java.io.IOException;
import java.util.Arrays;

/**
 * Class BitSet provides a set of integers from 0 to a given upper bound. The
 * set elements are stored in a bitmap representation. The largest integer that
 * can be stored is <I>max</I>&minus;1, where <I>max</I> is a constructor
 * argument.
 * <P>
 * <I>Note:</I> Class BitSet is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 22-Oct-2018
 */
public class BitSet
	implements Streamable
	{

// Hidden data members.

	private int max;
	private int N;
	private int[] bitmap;

// Exported constructors.

	/**
	 * Construct a new uninitialized set. This constructor is for use only by
	 * object streaming.
	 */
	public BitSet()
		{
		}

	/**
	 * Construct a new empty set. The set can hold elements from 0 through
	 * <I>max</I>&minus;1 inclusive.
	 *
	 * @param  max  Maximum number of elements (&ge; 1).
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>max</TT> &lt; 1.
	 */
	public BitSet
		(int max)
		{
		if (max < 1)
			throw new IllegalArgumentException (String.format
				("BitSet(): max = %d illegal", max));
		this.max = max;
		this.N = (max + 31) >> 5;
		this.bitmap = new int [N + 32];
		// 32 extra padding elements = kludge to avert false sharing in
		// multithreaded programs.
		}

	/**
	 * Construct a new set that is a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public BitSet
		(BitSet set)
		{
		copy (set);
		}

// Exported operations.

	/**
	 * Determine if this set is empty.
	 *
	 * @return  True if this set is empty, false otherwise.
	 */
	public boolean isEmpty()
		{
		for (int i = 0; i < N; ++ i)
			if (bitmap[i] != 0) return false;
		return true;
		}

	/**
	 * Clear this set.
	 *
	 * @return  This set.
	 */
	public BitSet clear()
		{
		for (int i = 0; i < N; ++ i)
			bitmap[i] = 0;
		return this;
		}

	/**
	 * Change this set to be a copy of the given set.
	 *
	 * @param  set  Set to copy.
	 *
	 * @return  This set.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>set</TT> is null.
	 */
	public BitSet copy
		(BitSet set)
		{
		this.max = set.max;
		if (set.N == 0)
			{
			this.N = 0;
			this.bitmap = null;
			}
		else
			{
			if (this.N != set.N)
				{
				this.N = set.N;
				this.bitmap = new int [N + 32];
				}
			System.arraycopy (set.bitmap, 0, this.bitmap, 0, N);
			}
		return this;
		}

	/**
	 * Returns the number of elements in this set.
	 *
	 * @return  Number of elements.
	 */
	public int size()
		{
		int n = 0;
		for (int i = 0; i < N; ++ i)
			n += Integer.bitCount (bitmap[i]);
		return n;
		}

	/**
	 * Determine if this set contains the given element. If <TT>elem</TT> is not
	 * in the range 0 .. <I>max</I>&minus;1, false is returned.
	 *
	 * @param  elem  Element.
	 *
	 * @return  True if this set contains <TT>elem</TT>, false otherwise.
	 */
	public boolean contains
		(int elem)
		{
		int i = indexForElement (elem);
		return i >= 0 && (bitmap[i] & maskForElement (i, elem)) != 0;
		}

	/**
	 * Add the given element to this set. If <TT>elem</TT> is not in the range 0
	 * .. <I>max</I>&minus;1, this set is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet add
		(int elem)
		{
		int i = indexForElement (elem);
		if (i >= 0)
			bitmap[i] |= maskForElement (i, elem);
		return this;
		}

	/**
	 * Add all elements in the given range to this set. All elements from
	 * <TT>lb</TT> through <TT>ub</TT>&minus;1 inclusive are added to this set.
	 * If any element is not in the range 0 .. <I>max</I>&minus;1, that element
	 * is not added. If <TT>lb</TT> &ge; <TT>ub</TT>, this set is unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet add
		(int lb,
		 int ub)
		{
		for (int i = 0; i < N; ++ i)
			bitmap[i] |= maskForRange (i, lb, ub);
		return this;
		}

	/**
	 * Remove the given element from this set. If <TT>elem</TT> is not in the
	 * range 0 .. <I>max</I>&minus;1, this set is unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet remove
		(int elem)
		{
		int i = indexForElement (elem);
		if (i >= 0)
			bitmap[i] &= ~maskForElement (i, elem);
		return this;
		}

	/**
	 * Remove all elements in the given range from this set. All elements from
	 * <TT>lb</TT> through <TT>ub</TT>&minus;1 inclusive are removed from this
	 * set. If any element is not in the range 0 .. <I>max</I>&minus;1, that
	 * element is not removed. If <TT>lb</TT> &ge; <TT>ub</TT>, this set is
	 * unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet remove
		(int lb,
		 int ub)
		{
		for (int i = 0; i < N; ++ i)
			bitmap[i] &= ~maskForRange (i, lb, ub);
		return this;
		}

	/**
	 * Flip the given element. If this set contains <TT>elem</TT>, it is
	 * removed; if this set does not contain <TT>elem</TT>, it is added. If
	 * <TT>elem</TT> is not in the range 0 .. <I>max</I>&minus;1, this set is
	 * unchanged.
	 *
	 * @param  elem  Element.
	 *
	 * @return  This set.
	 */
	public BitSet flip
		(int elem)
		{
		int i = indexForElement (elem);
		if (i >= 0)
			bitmap[i] ^= maskForElement (i, elem);
		return this;
		}

	/**
	 * Flip all elements in the given range. All elements from <TT>lb</TT>
	 * through <TT>ub</TT>&minus;1, inclusive, are flipped. If this set contains
	 * such an element, it is removed; if this set does not contain such an
	 * element, it is added. If any element is not in the range 0 ..
	 * <I>max</I>&minus;1, that element is not flipped. If <TT>lb</TT> &ge;
	 * <TT>ub</TT>, this set is unchanged.
	 *
	 * @param  lb  Lower bound element (inclusive).
	 * @param  ub  Upper bound element (exclusive).
	 *
	 * @return  This set.
	 */
	public BitSet flip
		(int lb,
		 int ub)
		{
		for (int i = 0; i < N; ++ i)
			bitmap[i] ^= maskForRange (i, lb, ub);
		return this;
		}

	/**
	 * Determine if this set is a subset of the given set. This is so if every
	 * element of this set is also an element of the given set.
	 * <P>
	 * It is assumed that <I>max</I> is the same for both sets; if not, the
	 * behavior of the <TT>isSubsetOf()</TT> method is not specified.
	 *
	 * @param  set  Set.
	 *
	 * @return  True if this set is a subset of the given set, false otherwise.
	 */
	public boolean isSubsetOf
		(BitSet set)
		{
		for (int i = 0; i < N; ++ i)
			if ((this.bitmap[i] & set.bitmap[i]) != this.bitmap[i])
				return false;
		return true;
		}

	/**
	 * Change this set to be the union of itself and the given set. The union
	 * consists of all elements that appear in this set or the given set or
	 * both.
	 * <P>
	 * It is assumed that <I>max</I> is the same for both sets; if not, the
	 * behavior of the <TT>union()</TT> method is not specified.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet union
		(BitSet set)
		{
		for (int i = 0; i < N; ++ i)
			this.bitmap[i] |= set.bitmap[i];
		return this;
		}

	/**
	 * Change this set to be the intersection of itself and the given set. The
	 * intersection consists of all elements that appear in this set and the
	 * given set.
	 * <P>
	 * It is assumed that <I>max</I> is the same for both sets; if not, the
	 * behavior of the <TT>intersection()</TT> method is not specified.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet intersection
		(BitSet set)
		{
		for (int i = 0; i < N; ++ i)
			this.bitmap[i] &= set.bitmap[i];
		return this;
		}

	/**
	 * Change this set to be the difference of itself and the given set. The
	 * difference consists of all elements that appear in this set and not in
	 * the given set.
	 * <P>
	 * It is assumed that <I>max</I> is the same for both sets; if not, the
	 * behavior of the <TT>difference()</TT> method is not specified.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet difference
		(BitSet set)
		{
		for (int i = 0; i < N; ++ i)
			this.bitmap[i] &= ~set.bitmap[i];
		return this;
		}

	/**
	 * Change this set to be the symmetric difference of itself and the given
	 * set. The symmetric difference consists of all elements that appear in
	 * this set or the given set, but not both.
	 * <P>
	 * It is assumed that <I>max</I> is the same for both sets; if not, the
	 * behavior of the <TT>symmetricDifference()</TT> method is not specified.
	 *
	 * @param  set  Set.
	 *
	 * @return  This set.
	 */
	public BitSet symmetricDifference
		(BitSet set)
		{
		for (int i = 0; i < N; ++ i)
			this.bitmap[i] ^= set.bitmap[i];
		return this;
		}

	/**
	 * Perform the given action on each element in this set. For each element in
	 * this set from smallest to largest, the given <TT>action</TT>'s
	 * <TT>run()</TT> method is called, passing in the element.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  action  Action.
	 */
	public void forEachItemDo
		(IntAction action)
		{
		for (int j = 0; j < N; ++ j)
			{
			int off = j << 5;
			int b = bitmap[j];
			for (int i = 0; b != 0; ++ i, b >>>= 1)
				if ((b & 1) != 0)
					action.run (off + i);
			}
		}

	/**
	 * Perform the given action on each element in this set and return a result.
	 * For each element in this set from smallest to largest, the given
	 * <TT>action</TT>'s <TT>run()</TT> method is called, passing in the
	 * element. After all the elements have been processed, the given
	 * <TT>action</TT>'s <TT>result()</TT> method is called, and its result is
	 * returned.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>action</TT> adds elements to or removes
	 * elements from the set, the <TT>forEachItemDo()</TT> method's behavior is
	 * not specified.
	 *
	 * @param  <R>     Result data type.
	 * @param  action  Action.
	 *
	 * @return  Result of processing all the set elements.
	 */
	public <R> R forEachItemDo
		(IntActionResult<R> action)
		{
		for (int j = 0; j < N; ++ j)
			{
			int off = j << 5;
			int b = bitmap[j];
			for (int i = 0; b != 0; ++ i, b >>>= 1)
				if ((b & 1) != 0)
					action.run (off + i);
			}
		return action.result();
		}

	/**
	 * Evaluate the given predicate on, and possibly remove, each element in
	 * this set. For each element in this set from smallest to largest, the
	 * given <TT>predicate</TT>'s <TT>test()</TT> method is called, passing in
	 * the element. If the <TT>test()</TT> method returns true, the element is
	 * removed from this set.
	 * <P>
	 * <B><I>Warning:</I></B> If the <TT>predicate</TT> adds elements to or
	 * removes elements from the set, other than by returning true, the
	 * <TT>removeEachItemIf()</TT> method's behavior is not specified.
	 *
	 * @param  predicate  Predicate.
	 */
	public void removeEachItemIf
		(IntPredicate predicate)
		{
		for (int j = 0; j < N; ++ j)
			{
			int off = j << 5;
			int b = bitmap[j];
			for (int i = 0; b != 0; ++ i, b >>>= 1)
				if ((b & 1) != 0 && predicate.test (off + i))
					remove (off + i);
			}
		}

	/**
	 * Return an element in this set chosen uniformly at random.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Random element, or &minus;1 if this set is empty.
	 */
	public int randomElement
		(Random prng)
		{
		// i = number of random element.
		int n = size();
		if (n == 0) return -1;
		int i = prng.nextInt (n);

		// w = index of bitmap word containing the i-th element.
		int w = 0;
		int bitmap_w = bitmap[w];
		n = Integer.bitCount (bitmap_w);
		while (i >= n)
			{
			i -= n;
			++ w;
			bitmap_w = bitmap[w];
			n = Integer.bitCount (bitmap_w);
			}

		// Pick the i-th element in bitmap word w.
		int e = 0;
		for (;;)
			{
			if ((bitmap_w & (1 << e)) != 0)
				{
				if (i == 0) break;
				-- i;
				}
			++ e;
			}

		// Return chosen element.
		return w*32 + e;
		}

	/**
	 * Return an element not in this set chosen uniformly at random.
	 *
	 * @param  prng  Pseudorandom number generator.
	 *
	 * @return  Random non-element, or &minus;1 if this set is full.
	 */
	public int randomNonElement
		(Random prng)
		{
		// i = number of random non-element.
		int n = max - size();
		if (n == 0) return -1;
		int i = prng.nextInt (n);

		// w = index of bitmap word containing the i-th non-element.
		int w = 0;
		int bitmap_w = bitmap[w];
		n = 32 - Integer.bitCount (bitmap_w);
		while (i >= n)
			{
			i -= n;
			++ w;
			bitmap_w = bitmap[w];
			n = 32 - Integer.bitCount (bitmap_w);
			}

		// Pick the i-th non-element in bitmap word w.
		int e = 0;
		for (;;)
			{
			if ((bitmap_w & (1 << e)) == 0)
				{
				if (i == 0) break;
				-- i;
				}
			++ e;
			}

		// Return chosen non-element.
		return w*32 + e;
		}

	/**
	 * Find the smallest element in this set greater than or equal to the given
	 * value.
	 *
	 * @param  value  Starting value.
	 *
	 * @return  The smallest element in this set greater than or equal to
	 *          <TT>value</TT>, or &minus;1 if there is no such element.
	 */
	public int nextElement
		(int value)
		{
		int i, j;
		if (value < 0)
			{
			i = 0;
			j = 0;
			}
		else
			{
			i = value & 31;
			j = value >> 5;
			}
		while (j < N)
			{
			if ((bitmap[j] & (1 << i)) != 0)
				return (j << 5) | i;
			++ i;
			if (i == 32)
				{
				i = 0;
				++ j;
				}
			}
		return -1;
		}

	/**
	 * Store this set's elements in the given array. This set's elements are
	 * stored starting at array index 0. The order in which this set's elements
	 * appear in the array is not specified. The number of array elements stored
	 * is <TT>array.length</TT>. If this set contains fewer than
	 * <TT>array.length</TT> elements, the remaining array elements are set to
	 * 0.
	 *
	 * @param  array  Array in which to store elements.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 */
	public int[] toArray
		(int[] array)
		{
		return toArray (array, 0, array.length);
		}

	/**
	 * Store this set's elements in a portion of the given array. This set's
	 * elements are stored starting at array index <TT>off</TT>. The order in
	 * which this set's elements appear in the array is not specified. The
	 * number of array elements stored is <TT>len</TT>. If this set contains
	 * fewer than <TT>len</TT> elements, the remaining array elements are set to
	 * 0.
	 *
	 * @param  array  Array in which to store elements.
	 * @param  off    Index at which to store first element.
	 * @param  len    Number of elements to store.
	 *
	 * @return  The given array is returned.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>array</TT> is null.
	 * @exception  IndexOutOfBoundsException
	 *     (unchecked exception) Thrown if <TT>off</TT> &lt; 0, <TT>len</TT>
	 *     &lt; 0, or <TT>off</TT>+<TT>len</TT> &gt; <TT>array.length</TT>.
	 */
	public int[] toArray
		(int[] array,
		 int off,
		 int len)
		{
		if (off < 0 || len < 0 || off+len > array.length)
			{
			throw new IndexOutOfBoundsException();
			}
		int index = 0;
		int shift = 0;
		int n = 0;
		while (index < N && n < len)
			{
			if (((bitmap[index] >> shift) & 1) == 1)
				{
				array [off+n] = (index << 5) | shift;
				++ n;
				}
			++ shift;
			if (shift == 32)
				{
				shift = 0;
				++ index;
				}
			}
		if (n < len) Arrays.fill (array, off + n, off + len, 0);
		return array;
		}

	/**
	 * Write this object's fields to the given out stream.
	 *
	 * @param  out  Out stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void writeOut
		(OutStream out)
		throws IOException
		{
		out.writeInt (max);
		out.writeInt (N);
		for (int i = 0; i < N; ++ i)
			out.writeUnsignedInt (bitmap[i]);
		}

	/**
	 * Read this object's fields from the given in stream.
	 *
	 * @param  in  In stream.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 */
	public void readIn
		(InStream in)
		throws IOException
		{
		max = in.readInt();
		int newN = in.readInt();
		if (newN > 0)
			{
			if (N != newN)
				{
				N = newN;
				bitmap = new int [N + 32];
				}
			for (int i = 0; i < N; ++ i)
				bitmap[i] = in.readUnsignedInt();
			}
		else
			{
			N = 0;
			bitmap = null;
			}
		}

// Hidden operations.

	/**
	 * Returns the index in the bitmap array that contains <TT>elem</TT>, or
	 * &minus;1 if <TT>elem</TT> is out of bounds.
	 */
	private int indexForElement
		(int elem)
		{
		return (0 <= elem && elem < max) ? elem >> 5 : -1;
		}

	/**
	 * Returns a mask that has a 1 bit at position <TT>elem</TT> and 0 bits
	 * elsewhere, for bitmap word <TT>i</TT>.
	 */
	private static int maskForElement
		(int i,
		 int elem)
		{
		elem -= i << 5;
		return 0 <= elem && elem <= 31 ? 1 << elem : 0;
		}

	/**
	 * Returns a mask that has 1 bits at positions <TT>lb</TT> through
	 * <TT>ub</TT>&minus;1 inclusive and 0 bits elsewhere, for bitmap word
	 * <TT>i</TT>.
	 */
	private static int maskForRange
		(int i,
		 int lb,
		 int ub)
		{
		lb = Math.max (lb - (i << 5), 0);
		ub = Math.min (ub - (i << 5), 32);
		if (lb >= ub) return 0;
		int lbfill = lb == 32 ? 0xffffffff : (1 << lb) - 1;
		int ubfill = ub == 32 ? 0xffffffff : (1 << ub) - 1;
		return ubfill & ~lbfill;
		}

	}
