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
	
	[HideInInspector]
	public float steerInput;
	
	private void Start()
	{
		thread = new Thread(new ThreadStart(Receive));
		thread.Start();
	}

	private void Receive()
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
	
	private void OnApplicationQuit()
	{
		udp.Close();
		thread.Abort();
	}
}
