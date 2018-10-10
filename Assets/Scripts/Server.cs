using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class Server : MonoBehaviour {

	private byte[] _data = new byte[1024];
	private IPEndPoint _ipep;
	private UdpClient _sock;
	private IPEndPoint _sender;
	
	void Start () {
		try
		{
			_ipep = new IPEndPoint(IPAddress.Any, 22222);
			_sock = new UdpClient(_ipep);

			_sender = new IPEndPoint(IPAddress.Any, 0);
		}
		catch (Exception e)
		{
			Console.WriteLine(e);
			throw;
		}
	}

	void FixedUpdate () {
		try
		{
			_data = _sock.Receive(ref _sender);
		
			Debug.Log(_data);
		}
		catch (Exception e)
		{
			Console.WriteLine(e);
			throw;
		}
	}
}
