//******************************************************************************
//
// File:    SHA256Base.java
// Package: edu.rit.sha2
// Unit:    Class edu.rit.sha2.SHA256Base
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

package edu.rit.sha2;

import edu.rit.crst.Function;
import edu.rit.util.BigInt;

/**
 * Class SHA256Base is the abstract base class for the SHA-2 hash functions that
 * use a 32-bit word. It implements a 52-byte (416-bit) message. The
 * cryptographic function's input <I>A</I> is the first portion of the message
 * (256 bits); input <I>B</I> is the second portion of the message (160 bits);
 * output <I>C</I> is the digest (size determined by a subclass).
 * <P>
 * Class SHA256Base computes the digest of one 512-bit block consisting of the
 * message (416 bits) plus the padding (96 bits). The padding consists of one 1
 * bit, thirty-one 0 bits, and the message length (64 bits).
 *
 * @author  Alan Kaminsky
 * @version 25-Aug-2017
 */
public abstract class SHA256Base
	extends Function
	{

// Hidden data members.

	/**
	 * Initialization vector word 0.
	 */
	protected final int IV_0;

	/**
	 * Initialization vector word 1.
	 */
	protected final int IV_1;

	/**
	 * Initialization vector word 2.
	 */
	protected final int IV_2;

	/**
	 * Initialization vector word 3.
	 */
	protected final int IV_3;

	/**
	 * Initialization vector word 4.
	 */
	protected final int IV_4;

	/**
	 * Initialization vector word 5.
	 */
	protected final int IV_5;

	/**
	 * Initialization vector word 6.
	 */
	protected final int IV_6;

	/**
	 * Initialization vector word 7.
	 */
	protected final int IV_7;

	/**
	 * Round constants.
	 */
	private static final int[] K = new int[]
		{
		0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 
		0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 
		0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 
		0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 
		0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 
		0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 
		0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 
		0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 
		0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 
		0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 
		0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 
		0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 
		0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 
		0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3, 
		0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 
		0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2, 
		};

	/**
	 * Message schedule.
	 */
	private int[] W = new int [16];

// Exported constructors.

	/**
	 * Construct a new SHA-256 base cryptographic function.
	 *
	 * @param  IV_0  Initialization vector word 0.
	 * @param  IV_1  Initialization vector word 1.
	 * @param  IV_2  Initialization vector word 2.
	 * @param  IV_3  Initialization vector word 3.
	 * @param  IV_4  Initialization vector word 4.
	 * @param  IV_5  Initialization vector word 5.
	 * @param  IV_6  Initialization vector word 6.
	 * @param  IV_7  Initialization vector word 7.
	 */
	public SHA256Base
		(int IV_0,
		 int IV_1,
		 int IV_2,
		 int IV_3,
		 int IV_4,
		 int IV_5,
		 int IV_6,
		 int IV_7)
		{
		super();
		this.IV_0 = IV_0;
		this.IV_1 = IV_1;
		this.IV_2 = IV_2;
		this.IV_3 = IV_3;
		this.IV_4 = IV_4;
		this.IV_5 = IV_5;
		this.IV_6 = IV_6;
		this.IV_7 = IV_7;
		}

// Exported operations.

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
	 * Get the number of rounds for this cryptographic function.
	 *
	 * @return  Number of rounds.
	 */
	public int rounds()
		{
		return 64;
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
		int f = IV_5;
		int g = IV_6;
		int h = IV_7;

		int s;
		int tmp1;
		int tmp2;

		// Do 64 rounds.
		for (int t = 0; t <= 63; ++ t)
			{
			s = t & 15;

			// Compute round function.
			if (t >= 16)
				W[s] = sigma_1 (W[(s+14)&15]) + W[(s+9)&15] +
					sigma_0 (W[(s+1)&15]) + W[s];
			tmp1 = h + Sigma_1(e) + Ch(e,f,g) + K[t] + W[s];
			tmp2 = Sigma_0(a) + Maj(a,b,c);
			h = g;
			g = f;
			f = e;
			e = d + tmp1;
			d = c;
			c = b;
			b = a;
			a = tmp1 + tmp2;

			// Record digest for this round.
			recordDigest (C[t], a, b, c, d, e, f, g, h);
			}
		}


// Hidden operations.

	/**
	 * Record the current round digest.
	 *
	 * @param  dig  Digest.
	 * @param  a    Working variable a.
	 * @param  b    Working variable b.
	 * @param  c    Working variable c.
	 * @param  d    Working variable d.
	 * @param  e    Working variable e.
	 * @param  f    Working variable f.
	 * @param  g    Working variable g.
	 * @param  h    Working variable h.
	 */
	protected abstract void recordDigest
		(BigInt dig,
		 int a,
		 int b,
		 int c,
		 int d,
		 int e,
		 int f,
		 int g,
		 int h);

	/**
	 * The little functions.
	 */
	private static int Ch (int x, int y, int z)
		{
		return (x & y) ^ (~x & z);
		}

	private static int Maj (int x, int y, int z)
		{
		return (x & y) ^ (x & z) ^ (y & z);
		}

	private static int Sigma_0 (int x)
		{
		return Integer.rotateRight (x, 2) ^ Integer.rotateRight (x, 13) ^
			Integer.rotateRight (x, 22);
		}

	private static int Sigma_1 (int x)
		{
		return Integer.rotateRight (x, 6) ^ Integer.rotateRight (x, 11) ^
			Integer.rotateRight (x, 25);
		}

	private static int sigma_0 (int x)
		{
		return Integer.rotateRight (x, 7) ^ Integer.rotateRight (x, 18) ^
			(x >>> 3);
		}

	private static int sigma_1 (int x)
		{
		return Integer.rotateRight (x, 17) ^ Integer.rotateRight (x, 19) ^
			(x >>> 10);
		}

	}
