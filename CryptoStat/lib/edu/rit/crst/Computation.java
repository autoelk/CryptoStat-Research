//******************************************************************************
//
// File:    Computation.java
// Package: edu.rit.crst
// Unit:    Class edu.rit.crst.Computation
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

import edu.rit.gpu.Gpu;
import edu.rit.gpu.Kernel;
import edu.rit.gpu.Module;
import edu.rit.util.Map;
import java.io.IOException;

/**
 * Class Computation is the abstract base class for an object that performs a
 * computation on the GPU. Class Computation provides a cache of loaded GPU
 * modules and GPU kernel functions shared by all the Computation instances.
 *
 * @author  Alan Kaminsky
 * @version 25-Apr-2017
 */
public abstract class Computation
	{

// Hidden helper class.

	private static class Info
		{
		// GPU module object.
		public Module module;

		// Map from kernel interface to GPU kernel object.
		public Map<Class<?>,Kernel> kernelMap;

		public Info
			(Module module)
			{
			this.module = module;
			this.kernelMap = new Map<Class<?>,Kernel>();
			}
		}

// Hidden data members.

	// Map from module name to info object.
	private static Map<String,Info> infoMap = new Map<String,Info>();

// Exported constructors.

	/**
	 * Construct a new computation object.
	 */
	public Computation()
		{
		}

// Hidden operations.

	/**
	 * Get a GPU kernel for this computation object.
	 * <P>
	 * If the GPU kernel module (whose name is returned by the {@link
	 * #moduleName() moduleName()} method) is not loaded, the <TT>kernel()</TT>
	 * method loads the module; calls the protected {@link
	 * #initializeModule(edu.rit.gpu.Gpu,edu.rit.gpu.Module) initializeModule()}
	 * method to initialize the module; creates a kernel object that implements
	 * the given interface; and caches the module and kernel objects internally.
	 * The <TT>kernel()</TT> method retrieves the kernel object from the cache
	 * and returns the kernel object.
	 *
	 * @param  <K>   Kernel interface.
	 * @param  gpu   GPU accelerator.
	 * @param  intf  Kernel interface's class object.
	 *
	 * @return  Kernel object.
	 *
	 * @exception  IOException
	 *     Thrown if an I/O error occurred.
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	protected <K extends Kernel> K kernel
		(Gpu gpu,
		 Class<K> intf)
		throws IOException
		{
		synchronized (infoMap)
			{
			Module module = null;
			Kernel kernel = null;

			// Get information for module.
			String name = moduleName();
			Info info = infoMap.get (name);
			if (info == null)
				{
				module = gpu.getModule (name);
				initializeModule (gpu, module);
				info = new Info (module);
				infoMap.put (name, info);
				}
			else
				{
				module = info.module;
				}

			// Get kernel object.
			kernel = info.kernelMap.get (intf);
			if (kernel == null)
				{
				kernel = module.getKernel (intf);
				info.kernelMap.put (intf, kernel);
				}

			return (K) kernel;
			}
		}

	/**
	 * Get the GPU kernel module name for this computation object.
	 * <P>
	 * The <TT>moduleName()</TT> method must be overridden in a subclass.
	 *
	 * @return  Module name.
	 */
	protected abstract String moduleName();

	/**
	 * Initialize the given kernel module.
	 * <P>
	 * The base class <TT>initializeModule()</TT> method does nothing. A
	 * subclass can override this method to do something.
	 *
	 * @param  gpu     GPU accelerator.
	 * @param  module  GPU kernel module.
	 *
	 * @exception  GpuException
	 *     (unchecked exception) Thrown if a GPU error occurred.
	 */
	protected void initializeModule
		(Gpu gpu,
		 Module module)
		{
		}

	}
