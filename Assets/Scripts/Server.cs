﻿using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class Server : MonoBehaviour
{
	
	private static readonly object lockObject = new object();
	private static UdpClient udp;
	private Thread thread;
	
	[HideInInspector]
	public float steerInput;
	
	private void Start()
	{
		try
		{
			thread = new Thread(new ThreadStart(Receive));
			thread.Start();
		} 
		catch (Exception e)
		{
			Debug.Log(e);
			throw;
		}
	}

	private void Receive()
	{
		try 
		{
			udp = new UdpClient(6969);
			while (true)
			{	
				var remoteIpEndPoint = new IPEndPoint(IPAddress.Any, 0);
				var receiveBytes = udp.Receive(ref remoteIpEndPoint);

				lock (lockObject)
				{
					steerInput = BitConverter.ToSingle(receiveBytes, 0);
				}
			}
		} 
		catch (Exception e)
		{
			Debug.Log(e);
			throw;
		}
	}
	
	private void OnApplicationQuit()
	{
		try 
		{
			udp.Close();
			thread.Abort();
		} 
		catch (Exception e) 
		{
			Debug.Log(e);
			throw;
		}
	}
}
