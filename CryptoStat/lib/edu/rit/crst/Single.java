//******************************************************************************
//
// File:    Single.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Single
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
 * Class Single provides an object that generates input values for a
 * cryptographic {@linkplain Function Function}.
 * <P>
 * Class Single generates just one cryptographic function input value, given by
 * a constructor argument.
 *
 * @author  Alan Kaminsky
 * @version 26-Oct-2016
 */
public class Single
	extends Generator
	{

// Hidden data members.

	private String value;

// Exported constructors.

	/**
	 * Construct a new single input generator.
	 *
	 * @param  value  Input value (hexadecimal string).
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>value</TT> is null.
	 */
	public Single
		(String value)
		{
		super();
		if (value == null)
			throw new NullPointerException ("Single(): value is null");
		this.value = value;
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
		return String.format ("Single(%s)", value);
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "Single input";
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
		GpuBigIntArray V = new GpuBigIntArray (bitSize, 1);
		V.item[0].fromString (value);
		return V;
		}

	}
