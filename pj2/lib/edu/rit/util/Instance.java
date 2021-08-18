//******************************************************************************
//
// File:    Instance.java
// Package: edu.rit.util
// Unit:    Class edu.rit.util.Instance
//
// This Java source file is copyright (C) 2016 by Alan Kaminsky. All rights
// reserved. For further information, contact the author, Alan Kaminsky, at
// ark@cs.rit.edu.
//
// This Java source file is part of the Parallel Java 2 Library ("PJ2"). PJ2 is
// free software; you can redistribute it and/or modify it under the terms of
// the GNU General Public License as published by the Free Software Foundation;
// either version 3 of the License, or (at your option) any later version.
//
// PJ2 is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
// A PARTICULAR PURPOSE. See the GNU General Public License for more details.
//
// A copy of the GNU General Public License is provided in the file gpl.txt. You
// may also obtain a copy of the GNU General Public License on the World Wide
// Web at http://www.gnu.org/licenses/gpl.html.
//
//******************************************************************************

package edu.rit.util;

import java.lang.reflect.Constructor;
import java.lang.reflect.InvocationTargetException;

/**
 * Class Instance provides static methods for creating instances of classes.
 *
 * @author  Alan Kaminsky
 * @version 12-Oct-2016
 */
public class Instance
	{

// Prevent construction.

	private Instance()
		{
		}

// Exported operations.

	/**
	 * Create a new instance of a class as specified by the given string.
	 * Calling this method is equivalent to calling
	 * <TT>newInstance(s,false)</TT>. See the {@link
	 * #newInstance(String,boolean) newInstance(String,boolean)} method for
	 * further information.
	 *
	 * @param  s  Constructor expression string.
	 *
	 * @return  New instance.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> does not obey the required
	 *     syntax.
	 * @exception  ClassNotFoundException
	 *     Thrown if the given class cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if a suitable constructor cannot be found in the given class.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the given constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the given constructor throws an exception.
	 */
	public static Object newInstance
		(String s)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newInstance (s, false);
		}

	/**
	 * Create a new instance of a class as specified by the given string. The
	 * string must consist of a fully-qualified class name, a left parenthesis,
	 * zero or more comma-separated values, and a right parenthesis. However,
	 * commas inside nested parentheses are part of the current value. No
	 * whitespace is allowed.
	 * <P>
	 * The <TT>newInstance()</TT> method examines all the declared constructors
	 * of the class and attempts to match each constructor argument's data type
	 * to the corresponding value in the constructor expression, as follows:
	 * <UL>
	 * <P><LI>
	 * If the constructor argument is a class type, the value must consist of a
	 * (nested) valid constructor expression for that class.
	 * <P><LI>
	 * If the constructor argument is type <TT>int</TT>, <TT>byte</TT>, or
	 * <TT>short</TT>, the value must consist of (case insensitive) an optional
	 * sign (<TT>+</TT> or <TT>-</TT>); <TT>0B</TT> plus one or more base-2
	 * digits, or <TT>0X</TT> plus one or more base-16 digits, or <TT>0</TT>
	 * plus zero or more base-8 digits, or <TT>1</TT> through <TT>9</TT> plus
	 * zero or more base-10 digits. The value must be in the range of type
	 * <TT>int</TT>. For an argument of type <TT>byte</TT> or <TT>short</TT>,
	 * the integer value is converted to a byte value or a short value.
	 * <P><LI>
	 * If the constructor argument is type <TT>long</TT>, the value must consist
	 * of (case insensitive) an optional sign (<TT>+</TT> or <TT>-</TT>);
	 * <TT>0B</TT> plus one or more base-2 digits, or <TT>0X</TT> plus one or
	 * more base-16 digits, or <TT>0</TT> plus zero or more base-8 digits, or
	 * <TT>1</TT> through <TT>9</TT> plus zero or more base-10 digits; optional
	 * <TT>L</TT>. The value must be in the range of type <TT>long</TT>.
	 * <P><LI>
	 * If the constructor argument is type <TT>double</TT> or <TT>float</TT>,
	 * the value must be as defined in the {@link Double#valueOf(String)
	 * Double.valueOf()} method. For an argument of type <TT>float</TT>, the
	 * double value is converted to a float value.
	 * <P><LI>
	 * If the constructor argument is type <TT>String</TT>, the value may be
	 * anything.
	 * <P><LI>
	 * If the final constructor argument is type <TT>int[]</TT>,
	 * <TT>byte[]</TT>, or <TT>short[]</TT>, all remaining values must be
	 * integers as described above; these values are bundled into an integer
	 * array, a byte array, or a short array.
	 * <P><LI>
	 * If the final constructor argument is type <TT>long[]</TT>, all remaining
	 * values must be long integers as described above; these values are bundled
	 * into a long integer array.
	 * <P><LI>
	 * If the final constructor argument is type <TT>double[]</TT> or
	 * <TT>float[]</TT>, all remaining values must be doubles as described
	 * above; these values are bundled into a double array or a float array.
	 * <P><LI>
	 * If the final constructor argument is type <TT>String[]</TT>, all
	 * remaining values are bundled into a string array.
	 * <P><LI>
	 * Constructor arguments of other types are not supported.
	 * </UL>
	 * <P>
	 * If one constructor that matches the values is found, the
	 * <TT>newInstance()</TT> method invokes that constructor, passing in the
	 * provided values, and returns a reference to the newly-created instance.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the instance. This means the object's class
	 * and/or the class's pertinent constructor need not be public, and a new
	 * instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 * <P>
	 * If no constructor that matches the values is found, or if multiple
	 * constructors that match the values are found, the <TT>newInstance()</TT>
	 * method throws a NoSuchMethodException.
	 * <P>
	 * <I>Note:</I> To find the given class, the calling thread's context class
	 * loader is used.
	 *
	 * @param  s  Constructor expression string.
	 * @param  disableAccessChecks  True to disable access checks, false to
	 *                              perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  IllegalArgumentException
	 *     (unchecked exception) Thrown if <TT>s</TT> does not obey the required
	 *     syntax.
	 * @exception  ClassNotFoundException
	 *     Thrown if the given class cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if a suitable constructor cannot be found in the given class.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newInstance
		(String s,
		 boolean disableAccessChecks)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		// Verify no whitespace.
		if (s.matches (".*\\s.*")) syntaxError ("No whitespace allowed");

		// Break constructor expression into class name and argument list.
		int a = s.indexOf ('(');
		if (a == -1) syntaxError ("Missing (");
		if (a == 0) syntaxError ("Missing class name");
		if (s.charAt (s.length() - 1) != ')') syntaxError ("Missing )");
		String className = s.substring (0, a);
		String argList = s.substring (a + 1, s.length() - 1);

		// Get class.
		Class<?> theClass = Class.forName
			(className,
			 true,
			 Thread.currentThread().getContextClassLoader());

		// Break argument list into comma-separated arguments.
		AList<String> args = parseTokens (argList);
		int N = args.size();

		// Attempt to parse arguments as int, long, or double values.
		Integer[] intValues = new Integer [N];
		Long[] longValues = new Long [N];
		Double[] doubleValues = new Double [N];
		for (int i = 0; i < N; ++ i)
			{
			intValues[i] = parseInt (args.get(i));
			longValues[i] = parseLong (args.get(i));
			doubleValues[i] = parseDouble (args.get(i));
			}

		// Get constructors and their argument types.
		Constructor<?>[] ctors = theClass.getDeclaredConstructors();
		int NC = ctors.length;
		Class<?>[][] argTypes = new Class<?> [NC] [];
		for (int i = 0; i < NC; ++ i)
			argTypes[i] = ctors[i].getParameterTypes();

		// Find matching constructor(s).
		Object[][] argValues = new Object [NC] [];
		int count = 0;
		int index = 0;
		for (int i = 0; i < NC; ++ i)
			{
			argValues[i] = matchConstructor (argTypes[i], args, intValues,
				longValues, doubleValues);
			if (argValues[i] != null)
				{
				++ count;
				index = i;
				}
			}

		// Found one suitable constructor.
		if (count == 1)
			{
			ctors[index].setAccessible (disableAccessChecks);
			return ctors[index].newInstance (argValues[index]);
			}

		// Found no suitable constructors.
		else if (count == 0)
			throw new NoSuchMethodException (String.format
				("Instance.newInstance(\"%s\"): Constructor not found",
				 s));

		// Found more than one suitable constructor.
		else
			throw new NoSuchMethodException (String.format
				("Instance.newInstance(\"%s\"): Multiple constructors found",
				 s));
		}

	/**
	 * Create a new instance of the class with the given name using the class's
	 * default constructor. Calling this method is equivalent to calling
	 * <TT>newDefaultInstance(className,false)</TT>. See the {@link
	 * #newDefaultInstance(String,boolean) newDefaultInstance(String,boolean)}
	 * method for further information.
	 * <P>
	 * <I>Note:</I> To find the class with the given name, the calling thread's
	 * context class loader is used.
	 *
	 * @param  className  Class name.
	 *
	 * @return  New instance.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class with the given name cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newDefaultInstance
		(String className)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newDefaultInstance (className, false);
		}

	/**
	 * Create a new instance of the class with the given name using the class's
	 * default constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the new instance. This means the object's
	 * class and/or the class's default constructor need not be public, and a
	 * new instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 * <P>
	 * <I>Note:</I> To find the class with the given name, the calling thread's
	 * context class loader is used.
	 *
	 * @param  className  Class name.
	 * @param  disableAccessChecks
	 *     True to disable access checks, false to perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  ClassNotFoundException
	 *     Thrown if the class with the given name cannot be found.
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static Object newDefaultInstance
		(String className,
		 boolean disableAccessChecks)
		throws
			ClassNotFoundException,
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		Class<?> theClass = Class.forName
			(className,
			 true,
			 Thread.currentThread().getContextClassLoader());
		return newDefaultInstance (theClass, disableAccessChecks);
		}

	/**
	 * Create a new instance of the given class using the class's default
	 * constructor. Calling this method is equivalent to calling
	 * <TT>newDefaultInstance(c,false)</TT>. See the {@link
	 * #newDefaultInstance(Class,boolean) newDefaultInstance(Class,boolean)}
	 * method for further information.
	 *
	 * @param  <T>  Class's data type.
	 * @param  c    Class.
	 *
	 * @return  New instance.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static <T> T newDefaultInstance
		(Class<T> c)
		throws
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return newDefaultInstance (c, false);
		}

	/**
	 * Create a new instance of the given class using the class's default
	 * constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing the new instance. This means the object's
	 * class and/or the class's default constructor need not be public, and a
	 * new instance will still be constructed. However, this also requires that
	 * either (a) a security manager is not installed, or (b) the security
	 * manager allows ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 *
	 * @param  <T>  Class's data type.
	 * @param  c    Class.
	 * @param  disableAccessChecks
	 *     True to disable access checks, false to perform access checks.
	 *
	 * @return  New instance.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the given class does not have a default constructor.
	 * @exception  InstantiationException
	 *     Thrown if an instance cannot be created because the given class is an
	 *     interface or an abstract class.
	 * @exception  IllegalAccessException
	 *     Thrown if an instance cannot be created because the calling method
	 *     does not have access to the class or constructor.
	 * @exception  InvocationTargetException
	 *     Thrown if the constructor throws an exception.
	 */
	public static <T> T newDefaultInstance
		(Class<T> c,
		 boolean disableAccessChecks)
		throws
			NoSuchMethodException,
			InstantiationException,
			IllegalAccessException,
			InvocationTargetException
		{
		return
			((Constructor<T>)(getDefaultConstructor (c, disableAccessChecks)))
				.newInstance();
		}

	/**
	 * Get the given class's default constructor. Calling this method is
	 * equivalent to calling <TT>getDefaultConstructor(c,false)</TT>. See the
	 * {@link #getDefaultConstructor(Class,boolean)
	 * getDefaultConstructor(Class,boolean)} method for further information.
	 *
	 * @param  c  Class.
	 *
	 * @return  Default (no-argument) constructor for the class.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the class does not have a default constructor.
	 */
	public static Constructor<?> getDefaultConstructor
		(Class<?> c)
		throws NoSuchMethodException
		{
		return getDefaultConstructor (c, false);
		}

	/**
	 * Get the given class's default constructor.
	 * <P>
	 * If the <TT>disableAccessChecks</TT> argument is true, access checks are
	 * suppressed when constructing an instance using the returned constructor.
	 * This means the object's class and/or the class's default constructor need
	 * not be public, and a new instance will still be constructed. However,
	 * this also requires that either (a) a security manager is not installed,
	 * or (b) the security manager allows
	 * ReflectPermission("suppressAccessChecks"). See the
	 * <TT>java.lang.reflect.Constructor.setAccessible()</TT> method for further
	 * information.
	 *
	 * @param  c  Class.
	 * @param  disableAccessChecks  True to disable access checks, false to
	 *                              perform access checks.
	 *
	 * @return  Default (no-argument) constructor for the class.
	 *
	 * @exception  NoSuchMethodException
	 *     Thrown if the class does not have a default constructor.
	 */
	public static Constructor<?> getDefaultConstructor
		(Class<?> c,
		 boolean disableAccessChecks)
		throws NoSuchMethodException
		{
		for (Constructor<?> ctor : c.getDeclaredConstructors())
			if (ctor.getParameterTypes().length == 0)
				{
				ctor.setAccessible (disableAccessChecks);
				return ctor;
				}
		throw new NoSuchMethodException (String.format
			("No such method: %s.<init>()", c.getName()));
		}

// Hidden operations.

	/**
	 * Throw an exception indicating a syntax error.
	 */
	private static void syntaxError
		(String msg)
		{
		throw new IllegalArgumentException (String.format
			("Instance.newInstance(): Syntax error: %s", msg));
		}

	/**
	 * Parse the given string into a series of tokens separated by commas.
	 * Inside nested parentheses, commas are part of the current token. Returns
	 * a list of the tokens.
	 */
	private static AList<String> parseTokens
		(String s)
		{
		AList<String> tokens = new AList<String>();
		int n = s.length();
		if (n == 0) return tokens;
		int parenLevel = 0;
		int i = 0;
		int j = 0;
		while (j < n)
			{
			char c = s.charAt (j);
			if (c == '(')
				{
				++ parenLevel;
				++ j;
				}
			else if (c == ')')
				{
				if (parenLevel == 0) syntaxError ("Too many )s");
				-- parenLevel;
				++ j;
				}
			else if (c == ',')
				{
				if (parenLevel == 0)
					{
					tokens.addLast (s.substring (i, j));
					i = j + 1;
					j = i;
					}
				else
					{
					++ j;
					}
				}
			else
				{
				++ j;
				}
			}
		if (parenLevel > 0) syntaxError ("Not enough )s");
		tokens.addLast (s.substring (i, j));
		return tokens;
		}

	/**
	 * Parse the given string as a value of type <TT>int</TT>. If successful,
	 * return the parsed value, otherwise return null.
	 */
	private static Integer parseInt
		(String s)
		{
		s = s.toUpperCase();

		String sign = "";
		if (s.charAt (0) == '+')
			{
			sign = "+";
			s = s.substring (1);
			}
		else if (s.charAt (0) == '-')
			{
			sign = "-";
			s = s.substring (1);
			}

		try
			{
			if (s.startsWith ("0B"))
				return Integer.valueOf (sign + s.substring (2), 2);
			else if (s.startsWith ("0X"))
				return Integer.valueOf (sign + s.substring (2), 16);
			else if (s.charAt (0) == '0')
				return Integer.valueOf (sign + s, 8);
			else
				return Integer.valueOf (sign + s, 10);
			}
		catch (NumberFormatException exc)
			{
			return null;
			}
		}

	/**
	 * Parse the given string as a value of type <TT>long</TT>. If successful,
	 * return the parsed value, otherwise return null.
	 */
	private static Long parseLong
		(String s)
		{
		s = s.toUpperCase();

		String sign = "";
		if (s.charAt (0) == '+')
			{
			sign = "+";
			s = s.substring (1);
			}
		else if (s.charAt (0) == '-')
			{
			sign = "-";
			s = s.substring (1);
			}

		if (s.charAt (s.length() - 1) == 'L')
			s = s.substring (0, s.length() - 1);

		try
			{
			if (s.startsWith ("0B"))
				return Long.valueOf (sign + s.substring (2), 2);
			else if (s.startsWith ("0X"))
				return Long.valueOf (sign + s.substring (2), 16);
			else if (s.charAt (0) == '0')
				return Long.valueOf (sign + s, 8);
			else
				return Long.valueOf (sign + s, 10);
			}
		catch (NumberFormatException exc)
			{
			return null;
			}
		}

	/**
	 * Parse the given string as a value of type <TT>double</TT>. If successful,
	 * return the parsed value, otherwise return null.
	 */
	private static Double parseDouble
		(String s)
		{
		try
			{
			return Double.valueOf (s);
			}
		catch (NumberFormatException exc)
			{
			return null;
			}
		}

	/**
	 * Match the given constructor argument types with the given values. If
	 * successful, return an array of objects representing the values, otherwise
	 * return null.
	 */
	private static Object[] matchConstructor
		(Class<?>[] argTypes,
		 AList<String> args,
		 Integer[] intValues,
		 Long[] longValues,
		 Double[] doubleValues)
		{
		if (argTypes.length > 0 && argTypes[argTypes.length-1].isArray())
			return matchConstructorVarargs
				(argTypes, args, intValues, longValues, doubleValues);
		else
			return matchConstructorNoVarargs
				(argTypes, args, intValues, longValues, doubleValues);
		}

	/**
	 * Match the given constructor argument types with the given values. The
	 * final argument type is an array type. If successful, return an array of
	 * objects representing the values, otherwise return null.
	 */
	private static Object[] matchConstructorVarargs
		(Class<?>[] argTypes,
		 AList<String> args,
		 Integer[] intValues,
		 Long[] longValues,
		 Double[] doubleValues)
		{
		int N = argTypes.length;
		if (args.size() < N - 1)
			return null;
		Object[] argValues = new Object [N];
		for (int i = 0; i < N - 1; ++ i)
			{
			argValues[i] = matchArgument (argTypes[i], args.get(i),
				intValues[i], longValues[i], doubleValues[i]);
			if (argValues[i] == null)
				return null;
			}
		argValues[N-1] = matchVarargs (argTypes[N-1], args, intValues,
			longValues, doubleValues, N - 1);
		if (argValues[N-1] == null)
			return null;
		return argValues;
		}

	/**
	 * Match the given constructor argument types with the given values. The
	 * final argument type is not an array type. If successful, return an array
	 * of objects representing the values, otherwise return null.
	 */
	private static Object[] matchConstructorNoVarargs
		(Class<?>[] argTypes,
		 AList<String> args,
		 Integer[] intValues,
		 Long[] longValues,
		 Double[] doubleValues)
		{
		int N = argTypes.length;
		if (args.size() != N)
			return null;
		Object[] argValues = new Object [N];
		for (int i = 0; i < N; ++ i)
			{
			argValues[i] = matchArgument (argTypes[i], args.get(i),
				intValues[i], longValues[i], doubleValues[i]);
			if (argValues[i] == null)
				return null;
			}
		return argValues;
		}

	/**
	 * Match the argument type at index i with the given argument values. If
	 * successful, return an object representing the argument value, otherwise
	 * return null.
	 */
	private static Object matchArgument
		(Class<?> argType,
		 String arg,
		 Integer intValue,
		 Long longValue,
		 Double doubleValue)
		{
		if (argType == Integer.TYPE && intValue != null)
			return intValue;
		else if (argType == Long.TYPE && longValue != null)
			return longValue;
		else if (argType == Double.TYPE && doubleValue != null)
			return doubleValue;
		else if (argType == Float.TYPE && doubleValue != null)
			return new Float (doubleValue);
		else if (argType == Byte.TYPE && intValue != null)
			return new Byte ((byte) intValue.intValue());
		else if (argType == Short.TYPE && intValue != null)
			return new Short ((short) intValue.intValue());
		else if (argType == String.class)
			return arg;
		else
			return null;
		}

	/**
	 * Match the final argument array type at index i with the given argument
	 * values. If successful, return an object representing the argument value,
	 * otherwise return null.
	 */
	private static Object matchVarargs
		(Class<?> argType,
		 AList<String> args,
		 Integer[] intValues,
		 Long[] longValues,
		 Double[] doubleValues,
		 int i)
		{
		if (argType == int[].class && allTypeInt (intValues, i))
			return toIntArray (intValues, i);
		else if (argType == long[].class && allTypeLong (longValues, i))
			return toLongArray (longValues, i);
		else if (argType == double[].class && allTypeDouble (doubleValues, i))
			return toDoubleArray (doubleValues, i);
		else if (argType == float[].class && allTypeDouble (doubleValues, i))
			return toFloatArray (doubleValues, i);
		else if (argType == byte[].class && allTypeInt (intValues, i))
			return toByteArray (intValues, i);
		else if (argType == short[].class && allTypeInt (intValues, i))
			return toShortArray (intValues, i);
		else if (argType == String[].class)
			return toStringArray (args, i);
		else
			return null;
		}

	/**
	 * Determine whether all remaining values starting at index i are type int.
	 */
	private static boolean allTypeInt
		(Integer[] intValues,
		 int i)
		{
		while (i < intValues.length)
			if (intValues[i++] == null)
				return false;
		return true;
		}

	/**
	 * Determine whether all remaining values starting at index i are type long.
	 */
	private static boolean allTypeLong
		(Long[] longValues,
		 int i)
		{
		while (i < longValues.length)
			if (longValues[i++] == null)
				return false;
		return true;
		}

	/**
	 * Determine whether all remaining values starting at index i are type
	 * double.
	 */
	private static boolean allTypeDouble
		(Double[] doubleValues,
		 int i)
		{
		while (i < doubleValues.length)
			if (doubleValues[i++] == null)
				return false;
		return true;
		}

	/**
	 * Return an integer array of all remaining values starting at index i.
	 */
	private static int[] toIntArray
		(Integer[] intValues,
		 int i)
		{
		int N = intValues.length - i;
		int[] rv = new int [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = intValues[i+j];
		return rv;
		}

	/**
	 * Return a long integer array of all remaining values starting at index i.
	 */
	private static long[] toLongArray
		(Long[] longValues,
		 int i)
		{
		int N = longValues.length - i;
		long[] rv = new long [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = longValues[i+j];
		return rv;
		}

	/**
	 * Return a double array of all remaining values starting at index i.
	 */
	private static double[] toDoubleArray
		(Double[] doubleValues,
		 int i)
		{
		int N = doubleValues.length - i;
		double[] rv = new double [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = doubleValues[i+j];
		return rv;
		}

	/**
	 * Return a float array of all remaining values starting at index i.
	 */
	private static float[] toFloatArray
		(Double[] doubleValues,
		 int i)
		{
		int N = doubleValues.length - i;
		float[] rv = new float [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = (float) doubleValues[i+j].doubleValue();
		return rv;
		}

	/**
	 * Return a byte array of all remaining values starting at index i.
	 */
	private static byte[] toByteArray
		(Integer[] intValues,
		 int i)
		{
		int N = intValues.length - i;
		byte[] rv = new byte [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = (byte) intValues[i+j].intValue();
		return rv;
		}

	/**
	 * Return a short integer array of all remaining values starting at index i.
	 */
	private static short[] toShortArray
		(Integer[] intValues,
		 int i)
		{
		int N = intValues.length - i;
		short[] rv = new short [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = (short) intValues[i+j].intValue();
		return rv;
		}

	/**
	 * Return a string array of all remaining values starting at index i.
	 */
	private static String[] toStringArray
		(AList<String> args,
		 int i)
		{
		int N = args.size() - i;
		String[] rv = new String [N];
		for (int j = 0; j < N; ++ j)
			rv[j] = args.get(i+j);
		return rv;
		}

// Unit test main program.

//	/**
//	 * Unit test main program.
//	 */
//	public static void main
//		(String[] args)
//		throws Exception
//		{
//		System.out.println (Instance.newInstance (args[0]));
//		}

	}
