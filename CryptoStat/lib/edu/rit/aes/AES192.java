//******************************************************************************
//
// File:    AES192.java
// Package: edu.rit.aes
// Unit:    Class edu.rit.aes.AES192
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

package edu.rit.aes;

import edu.rit.gpu.Gpu;
import java.io.IOException;

/**
 * Class AES192 implements the AES block cipher with a 192-bit key. The AES192
 * cryptographic function's input <I>A</I> is the plaintext (128 bits); input
 * <I>B</I> is the key (192 bits); output <I>C</I> is the ciphertext (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 20-Sep-2017
 */
public class AES192
	extends AESBase
	{

// Exported constructors.

	/**
	 * Construct a new AES-192 cryptographic function.
	 */
	public AES192()
		{
		super();
		}

// Exported operations.

	/**
	 * Get a constructor expression for this cryptographic function. The
	 * constructor expression can be passed to the {@link
	 * edu.rit.util.Instance#newInstance(String)
	 * edu.rit.util.Instance.newInstance()} method to construct an object that
	 * is the same as this cryptographic function.
	 *
	 * @return  Constructor expression.
	 */
	public String constructor()
		{
		return "edu.rit.aes.AES192()";
		}

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "AES-192 block cipher";
		}

	/**
	 * Get the bit size of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> bit size.
	 */
	public int B_bitSize()
		{
		return 192;
		}

	/**
	 * Get the number of rounds for this cryptographic function.
	 *
	 * @return  Number of rounds.
	 */
	public int rounds()
		{
		return 12;
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/aes/AES192.ptx";
		}

	}
