//******************************************************************************
//
// File:    OneOff.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.OneOff
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

package edu.rit.crst;

import edu.rit.util.GpuBigIntArray;

/**
 * Class OneOff provides an object that generates input values for a
 * cryptographic {@linkplain Function Function}.
 * <P>
 * Class OneOff generates <I>N</I>+1 input values, where <I>N</I> is a
 * constructor argument; the default <I>N</I> is the cryptographic function's
 * input bit size. The input values are 0, 1, 2, 4, 8, 16,
 * .&nbsp;.&nbsp;.&nbsp;, 2<SUP><I>N</I></SUP>&minus;1.
 * <P>
 * Feeding the OneOff generator into the {@linkplain Xor} generator results in a
 * sequence of input values where the second input value is the same as the
 * initial input value, with bit 0 flipped; the third input value is the same as
 * the initial input value, with bit 1 flipped; and so on.
 *
 * @author  Alan Kaminsky
 * @version 24-Oct-2017
 */
public class OneOff
	extends Generator
	{

// Hidden data members.

	private Integer N;

// Exported constructors.

	/**
	 * Construct a new one-off input generator. It will generate <I>N</I>+1
	 * input values, where <I>N</I> is the cryptographic function's input bit
	 * size.
	 */
	public OneOff()
		{
		super();
		}

	/**
	 * Construct a new one-off input generator with the given number of input
	 * values. It will generate <I>N</I>+1 input values, where <I>N</I> is the
	 * smaller of the argument <TT>N</TT> and the cryptographic function's input
	 * bit size.
	 *
	 * @param  N  Specifies number of input values.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>N</TT> &lt; 1.
	 */
	public OneOff
		(int N)
		{
		super();
		if (N < 1)
			throw new IllegalArgumentException (String.format
				("OneOff(): N = %d illegal", N));
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
		return (N == null) ? "OneOff()" : "OneOff("+N+")";
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "One-off inputs";
		}

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public GpuBigIntArray generate()
		{
		int bitSize = (gen_A) ? func.A_bitSize() : func.B_bitSize();
		int len = (N == null) ? bitSize : Math.min (bitSize, N);
		GpuBigIntArray V = new GpuBigIntArray (bitSize, len + 1);
		V.item[1] .increment();
		for (int i = 2; i <= len; ++ i)
			V.item[i] .assign (V.item[i-1]) .leftShift (1);
		return V;
		}

	}
