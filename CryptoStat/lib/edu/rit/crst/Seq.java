//******************************************************************************
//
// File:    Seq.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Seq
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
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

package edu.rit.crst;

import edu.rit.util.GpuBigIntArray;

/**
 * Class Seq provides an object that generates input values for a cryptographic
 * {@linkplain Function Function}.
 * <P>
 * Class Seq generates <I>N</I> cryptographic function input values, where
 * <I>N</I> is a constructor argument. The input values are 0, 1, 2,
 * .&nbsp;.&nbsp;.&nbsp;.&nbsp;, <I>N</I>&minus;1.
 * <P>
 * Feeding the Seq generator into the {@linkplain Add} generator results in a
 * sequence of input values that starts at an initial value and increases from
 * there.
 *
 * @author  Alan Kaminsky
 * @version 16-Nov-2016
 */
public class Seq
	extends Generator
	{

// Hidden data members.

	private int N;

// Exported constructors.

	/**
	 * Construct a new sequential input generator.
	 *
	 * @param  N  Number of input values to generate.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>N</TT> &lt; 1.
	 */
	public Seq
		(int N)
		{
		super();
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("Seq(): N = %d illegal", N));
		this.N = N;
		}

// Exported operations.

	/**
	 * Get a constructor expression for this input generator. The constructor
	 * expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this input generator.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return String.format ("Seq(%d)", N);
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "Sequential inputs";
		}

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public GpuBigIntArray generate()
		{
		int bitSize = gen_A ? func.A_bitSize() : func.B_bitSize();
		GpuBigIntArray V = new GpuBigIntArray (bitSize, N);
		for (int i = 1; i < N; ++ i)
			V.item[i] .assign (V.item[i-1]) .increment();
		return V;
		}

	}
