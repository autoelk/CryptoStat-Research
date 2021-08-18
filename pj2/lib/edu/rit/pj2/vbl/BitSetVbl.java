//******************************************************************************
//
// File:    BitSetVbl.java
// Package: edu.rit.pj2.vbl
// Unit:    Class edu.rit.pj2.vbl.BitSetVbl
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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
import edu.rit.util.BitSet;
import java.io.IOException;

/**
 * Class BitSetVbl provides a reduction variable for a set of integers from 0 to
 * a given upper bound shared by multiple threads executing a {@linkplain
 * edu.rit.pj2.ParallelStatement ParallelStatement}. Class BitSetVbl is a
 * {@linkplain Tuple} wrapping an instance of class {@linkplain BitSet}, which
 * is stored in the {@link #bitset bitset} field.
 * <P>
 * Class BitSetVbl supports the <I>parallel reduction</I> pattern. Each thread
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
 * <LI>Minimum size -- Class {@linkplain BitSetVbl.MinSize}
 * <LI>Maximum size -- Class {@linkplain BitSetVbl.MaxSize}
 * <LI>Set union -- Class {@linkplain BitSetVbl.Union}
 * <LI>Set intersection -- Class {@linkplain BitSetVbl.Intersection}
 * </UL>
 *
 * @author  Alan Kaminsky
 * @version 12-Oct-2016
 */
public class BitSetVbl
	extends Tuple
	implements Vbl
	{

// Exported data members.

	/**
	 * The bitset itself.
	 */
	public BitSet bitset;

// Exported constructors.

	/**
	 * Construct a new bitset reduction variable wrapping an empty bitset.
	 */
	public BitSetVbl()
		{
		this.bitset = new BitSet();
		}

	/**
	 * Construct a new bitset reduction variable wrapping the given bitset.
	 */
	public BitSetVbl
		(BitSet bitset)
		{
		this.bitset = bitset;
		}

// Exported operations.

	/**
	 * Create a clone of this shared variable.
	 *
	 * @return  The cloned object.
	 */
	public Object clone()
		{
		BitSetVbl vbl = (BitSetVbl) super.clone();
		if (this.bitset != null)
			vbl.bitset = new BitSet (this.bitset);
		return vbl;
		}

	/**
	 * Set this shared variable to the given shared variable.
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
		this.bitset.copy (((BitSetVbl)vbl).bitset);
		}

	/**
	 * Reduce the given bitset into this shared variable. This shared variable's
	 * {@link #bitset bitset} field and the <TT>bitset</TT> argument are
	 * combined together using this shared variable's reduction operation, and
	 * the result is stored in the {@link #bitset bitset} field.
	 * <P>
	 * The base class <TT>reduce()</TT> method throws an exception. The
	 * reduction operation must be defined in a subclass's <TT>reduce()</TT>
	 * method.
	 *
	 * @param  bitset  Bitset.
	 */
	public void reduce
		(BitSet bitset)
		{
		throw new UnsupportedOperationException
			("reduce() not defined in base class BitSetVbl; use a subclass");
		}

	/**
	 * Reduce the given shared variable into this shared variable. The two
	 * variables are combined together using this shared variable's reduction
	 * operation, and the result is stored in this shared variable.
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
		reduce (((BitSetVbl)vbl).bitset);
		}

	/**
	 * Write this bitset reduction variable to the given out stream.
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
		out.writeObject (bitset);
		}

	/**
	 * Read this bitset reduction variable from the given in stream.
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
		bitset = (BitSet) in.readObject();
		}

// Exported classes.

	/**
	 * Class BitSetVbl.MinSize provides a reduction variable for a set of
	 * integers from 0 to a given upper bound, where the reduction operation is
	 * to keep the set with the smallest size. The set elements are stored in a
	 * bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Oct-2016
	 */
	public static class MinSize
		extends BitSetVbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public MinSize()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public MinSize
			(BitSet bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given bitset into this shared variable. This shared
		 * variable's {@link #bitset bitset} field and the <TT>bitset</TT>
		 * argument are combined together using this shared variable's reduction
		 * operation, and the result is stored in the {@link #bitset bitset}
		 * field.
		 * <P>
		 * The BitSetVbl.MinSize class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} field to be a copy of the
		 * <TT>bitset</TT> argument if the latter's size is smaller.
		 *
		 * @param  bitset  Bitset.
		 */
		public void reduce
			(BitSet bitset)
			{
			if (bitset.size() < this.bitset.size())
				this.bitset.copy (bitset);
			}
		}

	/**
	 * Class BitSetVbl.MaxSize provides a reduction variable for a set of
	 * integers from 0 to a given upper bound, where the reduction operation is
	 * to keep the set with the largest size. The set elements are stored in a
	 * bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Oct-2016
	 */
	public static class MaxSize
		extends BitSetVbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public MaxSize()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public MaxSize
			(BitSet bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given bitset into this shared variable. This shared
		 * variable's {@link #bitset bitset} field and the <TT>bitset</TT>
		 * argument are combined together using this shared variable's reduction
		 * operation, and the result is stored in the {@link #bitset bitset}
		 * field.
		 * <P>
		 * The BitSetVbl.MaxSize class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} field to be a copy of the
		 * <TT>bitset</TT> argument if the latter's size is larger.
		 *
		 * @param  bitset  Bitset.
		 */
		public void reduce
			(BitSet bitset)
			{
			if (bitset.size() > this.bitset.size())
				this.bitset.copy (bitset);
			}
		}

	/**
	 * Class BitSetVbl.Union provides a reduction variable for a set of integers
	 * from 0 to a given upper bound, where the reduction operation is set
	 * union. The set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Oct-2016
	 */
	public static class Union
		extends BitSetVbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public Union()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public Union
			(BitSet bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given bitset into this shared variable. This shared
		 * variable's {@link #bitset bitset} field and the <TT>bitset</TT>
		 * argument are combined together using this shared variable's reduction
		 * operation, and the result is stored in the {@link #bitset bitset}
		 * field.
		 * <P>
		 * The BitSetVbl.Union class's <TT>reduce()</TT> method changes this
		 * variable's {@link #bitset bitset} field to be the set union of itself
		 * with the <TT>bitset</TT> argument.
		 *
		 * @param  bitset  Bitset.
		 */
		public void reduce
			(BitSet bitset)
			{
			this.bitset.union (bitset);
			}
		}

	/**
	 * Class BitSetVbl.Intersection provides a reduction variable for a set of
	 * integers from 0 to a given upper bound, where the reduction operation is
	 * set intersection. The set elements are stored in a bitmap representation.
	 *
	 * @author  Alan Kaminsky
	 * @version 12-Oct-2016
	 */
	public static class Intersection
		extends BitSetVbl
		{

		/**
		 * Construct a new bitset reduction variable wrapping an empty bitset.
		 */
		public Intersection()
			{
			super();
			}

		/**
		 * Construct a new bitset reduction variable wrapping the given bitset.
		 */
		public Intersection
			(BitSet bitset)
			{
			super (bitset);
			}

		/**
		 * Reduce the given bitset into this shared variable. This shared
		 * variable's {@link #bitset bitset} field and the <TT>bitset</TT>
		 * argument are combined together using this shared variable's reduction
		 * operation, and the result is stored in the {@link #bitset bitset}
		 * field.
		 * <P>
		 * The BitSetVbl.Intersection class's <TT>reduce()</TT> method changes
		 * this variable's {@link #bitset bitset} field to be the set
		 * intersection of itself with the <TT>bitset</TT> argument.
		 *
		 * @param  bitset  Bitset.
		 */
		public void reduce
			(BitSet bitset)
			{
			this.bitset.intersection (bitset);
			}
		}

	}
