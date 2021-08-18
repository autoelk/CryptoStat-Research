//******************************************************************************
//
// File:    IntArrayVbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.IntArrayVbl
//
// This Java source file is copyright (C) 2017 by Alan Kaminsky. All rights
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

package edu.rit.pj2.vbl;

import edu.rit.io.InStream;
import edu.rit.io.OutStream;
import edu.rit.pj2.Tuple;
import edu.rit.pj2.Vbl;
import java.io.IOException;

/**
 * Class IntArrayVbl provides an integer array reduction variable shared by
 * multiple threads executing a {@linkplain edu.rit.pj2.ParallelStatement
 * ParallelStatement}. An IntArrayVbl is also a {@linkplain Tuple}.
 * <P>
 * Class IntArrayVbl supports the <I>parallel reduction</I> pattern. Each thread
 * creates a thread-local copy of the shared variable by calling the {@link
 * edu.rit.pj2.Loop#threadLocal(Vbl) threadLocal()} method of class {@linkplain
 * edu.rit.pj2.Loop Loop} or the {@link edu.rit.pj2.Section#threadLocal(Vbl)
 * threadLocal()} method of class {@linkplain edu.rit.pj2.Section Section}. Each
 * thread performs operations on its own copy, without needing to synchronize
 * with the other threads. At the end of the parallel statement, the
 * thread-local copies are automatically <I>reduced</I> together, and the result
 * is stored in the original shared variable. The reduction is performed by the
 * shared variable's {@link #reduce(Vbl) reduce()} method.
 * <P>
 * The following subclasses provide various predefined reduction operations. You
 * can also define your own subclasses with customized reduction operations.
 * <UL>
 * <LI>Sum &mdash; Class {@linkplain IntArrayVbl.Sum}
 * <LI>Minimum &mdash; Class {@linkplain IntArrayVbl.Min}
 * <LI>Maximum &mdash; Class {@linkplain IntArrayVbl.Max}
 * <LI>Bitwise Boolean and &mdash; Class {@linkplain IntArrayVbl.And}
 * <LI>Bitwise Boolean or &mdash; Class {@linkplain IntArrayVbl.Or}
 * <LI>Bitwise Boolean exclusive-or &mdash; Class {@linkplain IntArrayVbl.Xor}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 31-Jan-2017
 */
public class IntArrayVbl
	extends Tuple
	implements Vbl
	{

// Exported data members.

	/**
	 * The shared integer array.
	 */
	public int[] item;

// Exported constructors.

	/**
	 * Construct a new shared integer array variable. The {@link #item item}
	 * field is a zero-length array.
	 */
	public IntArrayVbl()
		{
		item = new int [0];
		}

	/**
	 * Construct a new shared integer array variable with the given length. The
	 * {@link #item item} field's elements are initially 0.
	 *
	 * @param  len  Array length (&ge; 0).
	 */
	public IntArrayVbl
		(int len)
		{
		item = new int [len];
		}

// Exported operations.

	/**
	 * Clone this shared variable.
	 *
	 * @return  Clone of this shared variable.
	 */
	public Object clone()
		{
		IntArrayVbl vbl = (IntArrayVbl) super.clone();
		vbl.set (this);
		return vbl;
		}

	/**
	 * Set this shared variable to the given shared variable. This variable must
	 * be set to a deep copy of the given variable.
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void set
		(Vbl vbl)
		{
		this.item = (int[]) ((IntArrayVbl)vbl).item.clone();
		}

	/**
	 * Reduce the given integer into the given array element of this shared
	 * variable.
	 * <P>
	 * The {@link #item item} field's element at index <TT>i</TT> and the
	 * argument <TT>x</TT> are combined together using this shared variable's
	 * reduction operation, and the result is stored in the {@link #item item}
	 * field's element at index <TT>i</TT>.
	 * <P>
	 * The base class <TT>reduce()</TT> method throws an exception. The
	 * reduction operation must be defined in a subclass's <TT>reduce()</TT>
	 * method.
	 *
	 * @param  i  Array index.
	 * @param  x  Integer.
	 */
	public void reduce
		(int i,
		 int x)
		{
		throw new UnsupportedOperationException
			("reduce() not defined in base class IntArrayVbl; use a subclass");
		}

	/**
	 * Reduce the given integer array into this shared variable.
	 * <P>
	 * For each array index in the {@link #item item} field, the {@link #item
	 * item} field's element at that index and the <TT>array</TT>'s element at
	 * that index are combined together using this shared variable's reduction
	 * operation, and the result is stored in the {@link #item item} field's
	 * element at that index. If the <TT>array</TT> is longer than the {@link
	 * #item item} field, then the extra elements of the <TT>array</TT> are
	 * ignored. If the {@link #item item} field is longer than the
	 * <TT>array</TT>, then the extra elements of the {@link #item item} field
	 * are not updated.
	 *
	 * @param  array  Integer array.
	 */
	public void reduce
		(int[] array)
		{
		int len = Math.min (item.length, array.length);
		for (int i = 0; i < len; ++ i)
			reduce (i, array[i]);
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together using this shared variable's reduction
	 * operation, and the result is stored in this shared variable.
	 * <P>
	 * For each array index in the {@link #item item} field, this shared
	 * variable's {@link #item item} field's element at that index and the
	 * <TT>vbl</TT>'s {@link #item item} field's element at that index are
	 * combined together using this shared variable's reduction operation, and
	 * the result is stored in this shared variable's {@link #item item} field's
	 * element at that index. If the <TT>vbl</TT>'s {@link #item item} field is
	 * longer than this shared variable's {@link #item item} field, then the
	 * extra elements of the <TT>vbl</TT>'s {@link #item item} field are
	 * ignored. If this shared variable's {@link #item item} field is longer
	 * than the <TT>vbl</TT>'s {@link #item item} field, then the extra elements
	 * of this shared variable's {@link #item item} field are not updated.
	 *
	 * @param  vbl  Shared variable.
	 *
	 * @exception  ClassCastException
	 *     (unchecked exception) Thrown if the class of <TT>vbl</TT> is not
	 *     compatible with the class of this shared variable.
	 */
	public void reduce
		(Vbl vbl)
		{
		reduce (((IntArrayVbl)vbl).item);
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
		out.writeIntArray (item);
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
		item = in.readIntArray();
		}

// Exported subclasses.

	/**
	 * Class IntArrayVbl.Sum provides an integer array reduction variable, with
	 * addition as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class Sum
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public Sum()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public Sum
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the addition
		 * operation, and the result is stored in the {@link #item item} field's
		 * element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] += x;
			}
		}

	/**
	 * Class IntArrayVbl.Min provides an integer array reduction variable, with
	 * minimum as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class Min
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public Min()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public Min
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the minimum
		 * operation, and the result is stored in the {@link #item item} field's
		 * element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] = Math.min (item[i], x);
			}
		}

	/**
	 * Class IntArrayVbl.Max provides an integer array reduction variable, with
	 * maximum as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class Max
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public Max()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public Max
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the maximum
		 * operation, and the result is stored in the {@link #item item} field's
		 * element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] = Math.max (item[i], x);
			}
		}

	/**
	 * Class IntArrayVbl.And provides an integer array reduction variable, with
	 * bitwise Boolean and as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class And
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public And()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public And
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the bitwise Boolean
		 * and operation, and the result is stored in the {@link #item item}
		 * field's element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] &= x;
			}
		}

	/**
	 * Class IntArrayVbl.Or provides an integer array reduction variable, with
	 * bitwise Boolean or as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class Or
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public Or()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public Or
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the bitwise Boolean
		 * or operation, and the result is stored in the {@link #item item}
		 * field's element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] |= x;
			}
		}

	/**
	 * Class IntArrayVbl.Xor provides an integer array reduction variable, with
	 * bitwise Boolean exclusive-or as the reduction operation.
	 *
	 * @author  Alan Kaminsky
	 * @version 31-Jan-2017
	 */
	public static class Xor
		extends IntArrayVbl
		{
		/**
		 * Construct a new shared integer array variable. The {@link #item item}
		 * field is a zero-length array.
		 */
		public Xor()
			{
			super();
			}

		/**
		 * Construct a new shared integer array variable with the given length.
		 * The {@link #item item} field's elements are initially 0.
		 *
		 * @param  len  Array length (&ge; 0).
		 */
		public Xor
			(int len)
			{
			super (len);
			}

		/**
		 * Reduce the given integer into the given array element of this shared
		 * variable.
		 * <P>
		 * The {@link #item item} field's element at index <TT>i</TT> and the
		 * argument <TT>x</TT> are combined together using the bitwise Boolean
		 * exclusive-or operation, and the result is stored in the {@link #item
		 * item} field's element at index <TT>i</TT>.
		 *
		 * @param  i  Array index.
		 * @param  x  Integer.
		 */
		public void reduce
			(int i,
			 int x)
			{
			item[i] ^= x;
			}
		}

	}
