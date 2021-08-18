//******************************************************************************
//
// File:    SHA1.java
// Package: edu.rit.sha1
// Unit:    Class edu.rit.sha1.SHA1
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

package edu.rit.sha1;

import edu.rit.crst.Function;
import edu.rit.gpu.Gpu;
import edu.rit.util.BigInt;
import java.io.IOException;

/**
 * Class SHA1 implements the SHA-1 hash function with a 52-byte (416-bit)
 * message. The SHA1 cryptographic function's input <I>A</I> is the first
 * portion of the message (256 bits); input <I>B</I> is the second portion of
 * the message (160 bits); output <I>C</I> is the digest (160 bits).
 * <P>
 * Class SHA1 computes the digest of one 512-bit block consisting of the message
 * (416 bits) plus the padding (96 bits). The padding consists of one 1 bit,
 * thirty-one 0 bits, and the message length (64 bits).
 * <P>
 * <I>Note:</I> Class SHA1 is not multiple thread safe.
 *
 * @author  Alan Kaminsky
 * @version 23-Aug-2017
 */
public class SHA1
	extends Function
	{

// Hidden data members.

	// Initialization vector.
	private static final int IV_0 = 0x67452301;
	private static final int IV_1 = 0xefcdab89;
	private static final int IV_2 = 0x98badcfe;
	private static final int IV_3 = 0x10325476;
	private static final int IV_4 = 0xc3d2e1f0;

	// Message schedule.
	private int[] W = new int [16];

// Exported constructors.

	/**
	 * Construct a new SHA-1 cryptographic function.
	 */
	public SHA1()
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
		return "edu.rit.sha1.SHA1()";
		}

	/**
	 * Get a description of this cryptographic function.
	 *
	 * @return  Description.
	 */
	public String description()
		{
		return "SHA-1 hash function";
		}

	/**
	 * Get a description of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> description.
	 */
	public String A_description()
		{
		return "message bits 0-255";
		}

	/**
	 * Get the bit size of input <I>A</I> for this cryptographic function.
	 *
	 * @return  Input <I>A</I> bit size.
	 */
	public int A_bitSize()
		{
		return 256;
		}

	/**
	 * Get a description of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> description.
	 */
	public String B_description()
		{
		return "message bits 256-415";
		}

	/**
	 * Get the bit size of input <I>B</I> for this cryptographic function.
	 *
	 * @return  Input <I>B</I> bit size.
	 */
	public int B_bitSize()
		{
		return 160;
		}

	/**
	 * Get a description of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> description.
	 */
	public String C_description()
		{
		return "digest";
		}

	/**
	 * Get the bit size of output <I>C</I> for this cryptographic function.
	 *
	 * @return  Output <I>C</I> bit size.
	 */
	public int C_bitSize()
		{
		return 160;
		}

	/**
	 * Get the number of rounds for this cryptographic function.
	 *
	 * @return  Number of rounds.
	 */
	public int rounds()
		{
		return 80;
		}

	/**
	 * Evaluate this cryptographic function. The function is evaluated on inputs
	 * <I>A</I> and <I>B</I>, and the output of each round is stored in the
	 * <I>C</I> array.
	 * <P>
	 * <I>Note:</I> The <TT>evaluate()</TT> method is performed on the CPU in a
	 * single thread. It is intended as a cross-check for the GPU kernel.
	 *
	 * @param  A  Input <I>A</I>.
	 * @param  B  Input <I>B</I>.
	 * @param  C  Array of outputs <I>C</I>, indexed by round.
	 */
	public void evaluate
		(BigInt A,
		 BigInt B,
		 BigInt[] C)
		{
		// Store message block in message schedule W.
		A.unpackBigEndian (W, 0);
		B.unpackBigEndian (W, 8);
		W[13] = 0x80000000;
		W[14] = 0;
		W[15] = 416;

		// Initialize working variables.
		int a = IV_0;
		int b = IV_1;
		int c = IV_2;
		int d = IV_3;
		int e = IV_4;

		int s;
		int f_t;
		int K_t;
		int tmp;

		// Do 80 rounds.
		for (int t = 0; t <= 79; ++ t)
			{
			s = t & 15;

			// Compute round function.
			if (t >= 16)
				W[s] = Integer.rotateLeft
					(W[(s+13)&15] ^ W[(s+8)&15] ^ W[(s+2)&15] ^ W[s], 1);
			if (t <= 19)
				{
				f_t = (b & c) ^ (~b & d);
				K_t = 0x5a827999;
				}
			else if (t <= 39)
				{
				f_t = b ^ c ^ d;
				K_t = 0x6ed9eba1;
				}
			else if (t <= 59)
				{
				f_t = (b & c) ^ (b & d) ^ (c & d);
				K_t = 0x8f1bbcdc;
				}
			else
				{
				f_t = b ^ c ^ d;
				K_t = 0xca62c1d6;
				}
			tmp = Integer.rotateLeft (a, 5) + f_t + e + K_t + W[s];
			e = d;
			d = c;
			c = Integer.rotateLeft (b, 30);
			b = a;
			a = tmp;

			// Record digest for this round.
			C[t].value[4] = a + IV_0;
			C[t].value[3] = b + IV_1;
			C[t].value[2] = c + IV_2;
			C[t].value[1] = d + IV_3;
			C[t].value[0] = e + IV_4;
			}
		}

// Hidden operations.

	/**
	 * Get the GPU kernel module name for this computation object.
	 *
	 * @return  Module name.
	 */
	protected String moduleName()
		{
		return "edu/rit/sha1/SHA1.ptx";
		}
	}
