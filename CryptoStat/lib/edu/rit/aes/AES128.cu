//******************************************************************************
//
// File:    AES128.cu
// Unit:    AES128 CUDA functions
//
// This CUDA source file is copyright (C) 2017 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This CUDA source file is part of the CryptoStat Library ("CryptoStat").
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

/**
 * The CUDA implementation of the AES-128 block cipher. The <I>A</I> input is
 * the plaintext (128 bits). The <I>B</I> input is the key (128 bits). The
 * <I>C</I> output is the ciphertext (128 bits).
 *
 * @author  Alan Kaminsky
 * @version 09-Oct-2016
 */

#define NK 4

#include "AESBase.cu"
