using System;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using UnityEngine;

public class Server : MonoBehaviour
{
	
	private static readonly object lockObject = new object();
	private static UdpClient udp;
	private Thread thread;
	
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
					var x = BitConverter.ToSingle(receiveBytes, 0);
					if (x > 1)
					{
						CarController.SteerInput = 1;
					}
					else if(x < -1)
					{
						CarController.SteerInput = -1;
					}
					else
					{
						CarController.SteerInput = x;
					}
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
