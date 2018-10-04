using System.Net;
using System.Net.Sockets;
using System.Text;
using UnityEngine;

public class Server : MonoBehaviour {

	private byte[] _data = new byte[1024];
	private IPEndPoint ipep;
	private UdpClient sock;
	private IPEndPoint sender;
	
	void Start () {
		// connect to port
		ipep = new IPEndPoint(IPAddress.Any, 22222);
		sock = new UdpClient(ipep);

		sender = new IPEndPoint(IPAddress.Any, 0);

	}

	void FixedUpdate () {
		// inputs to controls
		_data = sock.Receive(ref sender);
		
		Debug.Log(_data);
	}
}
