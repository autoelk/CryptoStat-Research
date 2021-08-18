//******************************************************************************
//
// File:    Xor.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Xor
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

import edu.rit.util.BigInt;

/**
 * Class Xor provides an object that generates input values for a cryptographic
 * {@linkplain Function Function}.
 * <P>
 * An instance of class Xor is layered on top of another input generator. The
 * latter is specified as an argument to the Xor constructor, in the form of a
 * constructor expression. The Xor constructor uses the constructor expression
 * to create the underlying input generator. (For further information about
 * constructor expressions, see class {@linkplain edu.rit.util.Instance
 * edu.rit.util.Instance}.)
 * <P>
 * Class Xor invokes the underlying input generator to generate a series of
 * cryptographic function input values. Class Xor then exclusive-ors each input
 * value with a constant value specified as a constructor argument. This
 * constant serves as an "initial value" for the input sequence.
 *
 * @author  Alan Kaminsky
 * @version 16-Nov-2016
 */
public class Xor
	extends TransformGenerator
	{

// Hidden data members.

	private String init;
	private BigInt initBigInt;

// Exported constructors.

	/**
	 * Construct a new exclusive-or input generator.
	 *
	 * @param  init  Constant initial value (hexadecimal string).
	 * @param  ctor  Constructor expression for the underlying input generator.
	 *
	 * @exception  NullPointerException
	 *     (unchecked exception) Thrown if <TT>init</TT> is null. Thrown if
	 *     <TT>ctor</TT> is null.
	 */
	public Xor
		(String init,
		 String ctor)
		{
		super (ctor);
		if (init == null)
			throw new NullPointerException ("Xor(): init is null");
		this.init = init;
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
		super.setFunction (func, gen_A);
		if (gen_A)
			initBigInt = new BigInt (func.A_bitSize());
		else
			initBigInt = new BigInt (func.B_bitSize());
		initBigInt.fromString (init);
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
	public String constructor()
		{
		return "Xor(" + init + "," + ctor + ")";
		}

	/**
	 * Get a description of this input generator.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return gen.description() + ", xored with " + init;
		}

// Hidden operations.

	/**
	 * Transform the given input value.
	 *
	 * @param  value  Input value.
	 */
	protected void transform
		(BigInt value)
		{
		value.xor (initBigInt);
		}

	}
