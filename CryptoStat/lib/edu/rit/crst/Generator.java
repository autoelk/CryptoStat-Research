//******************************************************************************
//
// File:    Generator.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Generator
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
 * Class Generator is the abstract base class for an object that generates
 * input values for a cryptographic {@linkplain Function Function}.
 *
 * @author  Alan Kaminsky
 * @version 15-Nov-2016
 */
public abstract class Generator
	{

// Hidden data members.

	/**
	 * Cryptographic function object.
	 */
	protected Function func;

	/**
	 * True to generate A input values, false to generate B input values.
	 */
	protected boolean gen_A;

// Exported constructors.

	/**
	 * Construct a new cryptographic input generator.
	 */
	public Generator()
		{
		}

// Exported operations.

	/**
	 * Set the cryptographic function for this input generator.
	 *
	 * @param  func   Cryptographic function object.
	 * @param  gen_A  True to generate A input values, false to generate B input
	 *                values.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>func</TT> is null.
	 */
	public void setFunction
		(Function func,
		 boolean gen_A)
		{
		if (func == null)
			throw new NullPointerException
				("Generator.setFunction(): func is null");
		this.func = func;
		this.gen_A = gen_A;
		}

	/**
	 * Get a constructor expression for this input generator. The constructor
	 * expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this input generator.
	 *
	 * @return  Constructor expression.
	 */
	public abstract String constructor();

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public abstract String description();

	/**
	 * Generate a series of values for the designated input of the cryptographic
	 * function.
	 *
	 * @return  Array of input values.
	 */
	public abstract GpuBigIntArray generate();

	}
